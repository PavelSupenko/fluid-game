using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;

/// <summary>
/// CPU-based soft body simulation using Position Based Dynamics (PBD) + Jobs + Burst.
///
/// Instead of SPH fluid forces, particles are connected by springs that maintain
/// shape while allowing jelly-like deformation. Springs can break under stress,
/// allowing tearing.
///
/// PBD step per sub-step:
///   1. Build spatial hash (all alive particles)
///   2. Predict: velocity += gravity*dt, predictedPos = pos + vel*dt
///   3. Solve spring constraints on predictedPos (N iterations)
///   4. Collisions between bodies (spatial hash lookup)
///   5. Finalize: velocity = (predictedPos - pos)/dt, pos = predictedPos
///   6. Flask suction + boundaries
///   7. Sleep check
/// </summary>
public class FluidSimulationJobs : MonoBehaviour
{
    // ─── Container ───────────────────────────────────────────────
    [Header("Container Bounds")]
    public Vector2 containerMin = new Vector2(-4f, -3f);
    public Vector2 containerMax = new Vector2(4f, 4f);

    // ─── Particle Settings ───────────────────────────────────────
    [Header("Particle Grid (fallback if no image)")]
    public int gridWidth = 30;
    public int gridHeight = 20;
    public float particleSpacing = 0.15f;
    public float particleRadius = 0.05f;
    public float smoothingRadius = 0.4f; // Used for spatial hash cell size + wake radius

    // ─── PBD Soft Body Parameters ────────────────────────────────
    [Header("Soft Body (PBD)")]
    [Tooltip("Spring stiffness per solver iteration. 0 = loose jelly, 1 = rigid. " +
             "Effective stiffness = 1 - (1-stiffness)^iterations")]
    [Range(0.01f, 1f)]
    public float springStiffness = 0.3f;

    [Tooltip("Number of constraint solver iterations per sub-step. " +
             "More = stiffer, more accurate, but costs more CPU.")]
    [Range(1, 10)]
    public int constraintIterations = 4;

    [Tooltip("Collision repulsion between particles of different bodies")]
    public float collisionStiffness = 0.5f;

    [Tooltip("Collision radius — particles push apart when closer than this")]
    public float collisionRadius = 0.08f;

    // ─── Physics ─────────────────────────────────────────────────
    [Header("Physics")]
    public Vector2 gravity = new Vector2(0f, -9.81f);
    [Range(0f, 1f)]
    public float boundaryDamping = 0.3f;
    [Range(0.9f, 1f)]
    public float velocityDamping = 0.98f;
    public float timeScale = 1f;
    [Range(1, 8)]
    public int subSteps = 3;
    public float maxSpeed = 8f;

    // ─── Sleep System ────────────────────────────────────────────
    [Header("Sleep System")]
    public float sleepVelocityThreshold = 0.05f;
    public int sleepFramesRequired = 30;
    public int wakeBudgetPerFrame = 150;
    public float wakeRadius = 1.5f;
    public bool startSleeping = true;
    public int scanBudgetPerFrame = 400;

    // ─── Fluid Types (for color + flask targeting) ───────────────
    [Header("Fluid Types")]
    public FluidTypeDefinition[] fluidTypes = new FluidTypeDefinition[]
    {
        new FluidTypeDefinition { name = "Default", color = Color.white,
            density = 2f, viscosity = 6f, cohesion = 1f },
    };
    public bool uniformFluid = false;

    // ─── Flask (set by FlaskController) ──────────────────────────
    [HideInInspector] public float2 flaskPos;
    [HideInInspector] public bool flaskActive;
    [HideInInspector] public int flaskTargetType = -1;
    [HideInInspector] public float flaskRadius = 1f;
    [HideInInspector] public float flaskAbsorbRadius = 0.15f;
    [HideInInspector] public float flaskStrength = 80f;

    // ─── Public Accessors ────────────────────────────────────────
    public FluidParticle[] Particles { get; private set; }
    public int ParticleCount { get; private set; }
    public ComputeBuffer ParticleBuffer => particleBuffer;
    public int AwakeCount { get; private set; }
    public int SpringCount { get; private set; }
    public int BodyCount { get; private set; }
    public bool HasSoftBodies { get; private set; }

    // ─── Internal Native Data ────────────────────────────────────
    private NativeArray<ParticleData> particles;
    private NativeArray<float2> predictedPos;
    private NativeArray<int> sleepState;
    private NativeArray<int> sleepCounter;
    private NativeArray<int> bodyIndices;

    // Springs
    private NativeArray<SpringData> springs;
    private int springCount;

    // Spatial hash
    private NativeArray<int> cellCounts;
    private NativeArray<int> cellOffsets;
    private NativeArray<int> sortedIndices;
    private NativeArray<int> particleCellIndex;

    // Renderer bridge
    private ComputeBuffer particleBuffer;

    // Spatial hash dimensions
    private int hashGridW, hashGridH, hashGridTotal;
    private float cellSize;

    // ─── Burst-compatible structs ────────────────────────────────

    // Must produce 48-byte layout matching FluidParticle for GPU upload
    private struct ParticleData
    {
        public float2 position;     // 8
        public float2 velocity;     // 8
        public int typeIndex;       // 4
        public float density;       // 4 (unused in PBD, kept for struct layout)
        public float pressure;      // 4 (unused in PBD, kept for struct layout)
        public float alive;         // 4
        public float4 color;        // 16
    } // Total: 48 bytes

    private struct SpringData
    {
        public int particleA;
        public int particleB;
        public float restLength;
        public float breakThreshold;
        public int alive;
    }

    // ─── Lifecycle ───────────────────────────────────────────────

    void Awake()
    {
        var imageSource = GetComponent<ImageToFluid>();
        if (imageSource != null)
            imageSource.TryParseImage();

        if (imageSource != null && imageSource.IsReady)
            InitFromImage(imageSource);
        else
            SpawnParticles();

        // Process soft body mask
        var softBodySetup = GetComponent<SoftBodySetup>();
        if (softBodySetup != null && imageSource != null && imageSource.IsReady)
        {
            softBodySetup.Process(imageSource);
            if (softBodySetup.IsReady)
                HasSoftBodies = true;
        }

        InitNativeData();
    }

    void FixedUpdate()
    {
        float dt = (Time.fixedDeltaTime * timeScale) / subSteps;

        for (int step = 0; step < subSteps; step++)
            RunSimulationStep(dt);

        // Wake checks — run after simulation so spatial hash is fresh
        if (flaskActive)
        {
            WakeNearPoint(flaskPos, wakeRadius);
            WakeColumnAbove(flaskPos);
        }
        CascadeWake();
        DetectFloatingIslands();

        UploadToGPU();
    }

    void OnDestroy()
    {
        DisposeNative();
        particleBuffer?.Release();
    }

    // ─── Initialization ──────────────────────────────────────────

    void SpawnParticles()
    {
        ParticleCount = gridWidth * gridHeight;
        Particles = new FluidParticle[ParticleCount];
        float totalWidth = (gridWidth - 1) * particleSpacing;
        float totalHeight = (gridHeight - 1) * particleSpacing;
        float startX = -totalWidth * 0.5f;
        float startY = containerMax.y - 0.5f - totalHeight;
        int typeCount = fluidTypes.Length;
        int rowsPerType = Mathf.Max(1, gridHeight / typeCount);

        for (int y = 0; y < gridHeight; y++)
        {
            int typeIndex = Mathf.Min(y / rowsPerType, typeCount - 1);
            for (int x = 0; x < gridWidth; x++)
            {
                int i = y * gridWidth + x;
                Particles[i] = new FluidParticle
                {
                    position = new Vector2(startX + x * particleSpacing, startY + y * particleSpacing)
                             + new Vector2(UnityEngine.Random.Range(-0.01f, 0.01f),
                                           UnityEngine.Random.Range(-0.01f, 0.01f)),
                    velocity = Vector2.zero, typeIndex = typeIndex,
                    density = 0f, pressure = 0f, alive = 1f,
                    color = fluidTypes[typeIndex].color
                };
            }
        }
    }

    void InitFromImage(ImageToFluid source)
    {
        Particles = source.GeneratedParticles;
        ParticleCount = source.GeneratedParticleCount;
        fluidTypes = source.GeneratedFluidTypes;
        particleSpacing = source.ComputedSpacing;
        uniformFluid = true;
        startSleeping = true;

        float s = particleSpacing;
        smoothingRadius = s * 2.5f;
        particleRadius = s * 0.4f;
        collisionRadius = s * 0.8f;
        wakeRadius = smoothingRadius * 4f;
        subSteps = Mathf.Max(subSteps, 4);

        Debug.Log($"[FluidSimJobs] From image: {ParticleCount} particles, spacing={s:F4}");
    }

    void InitNativeData()
    {
        // Spatial hash grid
        cellSize = smoothingRadius;
        float containerW = containerMax.x - containerMin.x;
        float containerH = containerMax.y - containerMin.y;
        hashGridW = Mathf.CeilToInt(containerW / cellSize) + 1;
        hashGridH = Mathf.CeilToInt(containerH / cellSize) + 1;
        hashGridTotal = hashGridW * hashGridH;

        // Allocate native arrays
        particles = new NativeArray<ParticleData>(ParticleCount, Allocator.Persistent);
        predictedPos = new NativeArray<float2>(ParticleCount, Allocator.Persistent);
        sleepState = new NativeArray<int>(ParticleCount, Allocator.Persistent);
        sleepCounter = new NativeArray<int>(ParticleCount, Allocator.Persistent);

        cellCounts = new NativeArray<int>(hashGridTotal, Allocator.Persistent);
        cellOffsets = new NativeArray<int>(hashGridTotal, Allocator.Persistent);
        sortedIndices = new NativeArray<int>(ParticleCount, Allocator.Persistent);
        particleCellIndex = new NativeArray<int>(ParticleCount, Allocator.Persistent);

        // Copy managed → native
        for (int i = 0; i < ParticleCount; i++)
        {
            var p = Particles[i];
            particles[i] = new ParticleData
            {
                position = new float2(p.position.x, p.position.y),
                velocity = float2.zero,
                typeIndex = p.typeIndex,
                density = 0, pressure = 0,
                alive = p.alive,
                color = new float4(p.color.r, p.color.g, p.color.b, p.color.a)
            };
            sleepState[i] = (startSleeping && p.alive > 0.5f) ? 1 : 0;
            sleepCounter[i] = startSleeping ? sleepFramesRequired : 0;
        }

        // ── Soft body data ──
        var softBodySetup = GetComponent<SoftBodySetup>();
        if (HasSoftBodies && softBodySetup != null && softBodySetup.IsReady)
        {
            bodyIndices = new NativeArray<int>(ParticleCount, Allocator.Persistent);
            for (int i = 0; i < ParticleCount; i++)
                bodyIndices[i] = softBodySetup.BodyIndices[i];

            springCount = softBodySetup.SpringCount;
            SpringCount = springCount;
            BodyCount = softBodySetup.BodyCount;

            springs = new NativeArray<SpringData>(Mathf.Max(1, springCount), Allocator.Persistent);
            for (int i = 0; i < springCount; i++)
            {
                var s = softBodySetup.Springs[i];
                springs[i] = new SpringData
                {
                    particleA = s.particleA, particleB = s.particleB,
                    restLength = s.restLength, breakThreshold = s.breakThreshold,
                    alive = s.alive
                };
            }
            Debug.Log($"[FluidSimJobs] Soft bodies: {BodyCount} bodies, {SpringCount} springs");
        }
        else
        {
            bodyIndices = new NativeArray<int>(ParticleCount, Allocator.Persistent);
            springs = new NativeArray<SpringData>(1, Allocator.Persistent);
            springCount = 0; SpringCount = 0; BodyCount = 0;
        }

        particleBuffer = new ComputeBuffer(ParticleCount, 48);
        UploadToGPU();

        AwakeCount = startSleeping ? 0 : ParticleCount;
        Debug.Log($"[FluidSimJobs] Init: {ParticleCount} particles, grid {hashGridW}x{hashGridH}");
    }

    // ═════════════════════════════════════════════════════════════
    //  PBD SIMULATION STEP
    // ═════════════════════════════════════════════════════════════

    void RunSimulationStep(float dt)
    {
        // 1. Build spatial hash (ALL alive particles — sleeping + awake)
        new AssignCellsJob
        {
            particles = particles, particleCellIndex = particleCellIndex,
            cellSize = cellSize,
            containerMin = new float2(containerMin.x, containerMin.y),
            gridW = hashGridW, gridH = hashGridH
        }.Schedule(ParticleCount, 128).Complete();

        new BuildGridJob
        {
            particleCellIndex = particleCellIndex,
            cellCounts = cellCounts, cellOffsets = cellOffsets,
            sortedIndices = sortedIndices,
            particleCount = ParticleCount, gridTotal = hashGridTotal
        }.Run();

        // 2. Predict positions: vel += gravity*dt, predicted = pos + vel*dt
        new PredictJob
        {
            particles = particles, predictedPos = predictedPos,
            sleepState = sleepState,
            gravity = new float2(gravity.x, gravity.y), dt = dt
        }.Schedule(ParticleCount, 256).Complete();

        // 3. Solve constraints: springs + collisions interleaved (N iterations)
        for (int iter = 0; iter < constraintIterations; iter++)
        {
            // Springs hold shape
            new SpringSolveJob
            {
                springs = springs, predictedPos = predictedPos,
                particles = particles, sleepState = sleepState,
                springCount = springCount, stiffness = springStiffness
            }.Run();

            // Collision between ALL particles — prevents overlap.
            // Works for inter-body AND intra-body (broken-off chunks).
            new CollisionJob
            {
                particles = particles, predictedPos = predictedPos,
                sleepState = sleepState,
                sortedIndices = sortedIndices,
                cellCounts = cellCounts, cellOffsets = cellOffsets,
                cellSize = cellSize,
                containerMin = new float2(containerMin.x, containerMin.y),
                gridW = hashGridW, gridH = hashGridH,
                collisionRadius = collisionRadius, collisionStiffness = collisionStiffness
            }.Schedule(ParticleCount, 128).Complete();
        }

        // 5. Finalize: update velocity from position change, apply damping,
        //    flask suction, boundaries
        new FinalizeJob
        {
            particles = particles, predictedPos = predictedPos,
            sleepState = sleepState,
            dt = dt, velocityDamping = velocityDamping,
            maxSpeed = maxSpeed, particleRadius = particleRadius,
            boundaryDamping = boundaryDamping,
            containerMin = new float2(containerMin.x, containerMin.y),
            containerMax = new float2(containerMax.x, containerMax.y),
            flaskActive = flaskActive, flaskPos = flaskPos,
            flaskTargetType = flaskTargetType,
            flaskRadius = flaskRadius, flaskAbsorbRadius = flaskAbsorbRadius,
            flaskStrength = flaskStrength
        }.Schedule(ParticleCount, 128).Complete();

        // 6. Break overstretched springs
        if (springCount > 0)
        {
            new SpringBreakJob
            {
                springs = springs, particles = particles,
                springCount = springCount
            }.Run();
        }

        // 7. Sleep check
        new SleepCheckJob
        {
            particles = particles, sleepState = sleepState,
            sleepCounter = sleepCounter,
            sleepVelThresholdSqr = sleepVelocityThreshold * sleepVelocityThreshold,
            sleepFramesRequired = sleepFramesRequired
        }.Schedule(ParticleCount, 256).Complete();
    }

    // ═════════════════════════════════════════════════════════════
    //  WAKE / SLEEP LOGIC (main thread)
    // ═════════════════════════════════════════════════════════════

    private int cascadeScanCursor = 0;

    void CascadeWake()
    {
        int wakeBudget = wakeBudgetPerFrame;
        int scanBudget = scanBudgetPerFrame;
        float cascadeRadiusSqr = smoothingRadius * smoothingRadius * 4f;
        float movingThresholdSqr = sleepVelocityThreshold * sleepVelocityThreshold * 4f;

        for (int scanned = 0; scanned < scanBudget && wakeBudget > 0; scanned++)
        {
            int i = cascadeScanCursor;
            cascadeScanCursor = (cascadeScanCursor + 1) % ParticleCount;

            if (sleepState[i] != 0 || particles[i].alive < 0.5f) continue;
            if (math.lengthsq(particles[i].velocity) < movingThresholdSqr) continue;

            float2 pos = particles[i].position;
            int2 cell = CellCoord(pos);

            for (int dx = -1; dx <= 1 && wakeBudget > 0; dx++)
            for (int dy = -1; dy <= 1 && wakeBudget > 0; dy++)
            {
                int2 nc = cell + new int2(dx, dy);
                if (nc.x < 0 || nc.x >= hashGridW || nc.y < 0 || nc.y >= hashGridH) continue;

                int ci = nc.y * hashGridW + nc.x;
                int start = cellOffsets[ci];
                int count = cellCounts[ci];

                for (int s = 0; s < count && wakeBudget > 0; s++)
                {
                    int j = sortedIndices[start + s];
                    if (sleepState[j] != 1) continue;
                    if (math.lengthsq(particles[j].position - pos) < cascadeRadiusSqr)
                    {
                        sleepState[j] = 0; sleepCounter[j] = 0; wakeBudget--;
                    }
                }
            }
        }
    }

    void WakeNearPoint(float2 point, float radius)
    {
        int wakeBudget = wakeBudgetPerFrame;
        float radiusSqr = radius * radius;

        int2 minCell = CellCoord(point - new float2(radius, radius));
        int2 maxCell = CellCoord(point + new float2(radius, radius));

        for (int cx = minCell.x; cx <= maxCell.x && wakeBudget > 0; cx++)
        for (int cy = minCell.y; cy <= maxCell.y && wakeBudget > 0; cy++)
        {
            if (cx < 0 || cx >= hashGridW || cy < 0 || cy >= hashGridH) continue;
            int ci = cy * hashGridW + cx;
            int start = cellOffsets[ci]; int count = cellCounts[ci];

            for (int s = 0; s < count && wakeBudget > 0; s++)
            {
                int i = sortedIndices[start + s];
                if (sleepState[i] == 0 || particles[i].alive < 0.5f) continue;

                bool typeMatch = (flaskTargetType < 0) || (particles[i].typeIndex == flaskTargetType);
                if (!typeMatch) continue;

                if (math.lengthsq(particles[i].position - point) < radiusSqr)
                { sleepState[i] = 0; sleepCounter[i] = 0; wakeBudget--; }
            }
        }
    }

    void WakeColumnAbove(float2 point)
    {
        int wakeBudget = wakeBudgetPerFrame;
        float halfWidth = smoothingRadius * 1.5f;

        int2 colMin = CellCoord(new float2(point.x - halfWidth, point.y));
        int2 colMax = CellCoord(new float2(point.x + halfWidth, containerMax.y));

        for (int cy = colMin.y; cy <= colMax.y && wakeBudget > 0; cy++)
        for (int cx = colMin.x; cx <= colMax.x && wakeBudget > 0; cx++)
        {
            if (cx < 0 || cx >= hashGridW || cy < 0 || cy >= hashGridH) continue;
            int ci = cy * hashGridW + cx;
            int start = cellOffsets[ci]; int count = cellCounts[ci];

            for (int s = 0; s < count && wakeBudget > 0; s++)
            {
                int i = sortedIndices[start + s];
                if (sleepState[i] != 1 || particles[i].alive < 0.5f) continue;

                bool typeMatch = (flaskTargetType < 0) || (particles[i].typeIndex == flaskTargetType);
                if (!typeMatch) continue;

                float dx2 = particles[i].position.x - point.x;
                if (dx2 * dx2 < halfWidth * halfWidth)
                { sleepState[i] = 0; sleepCounter[i] = 0; wakeBudget--; }
            }
        }
    }

    void DetectFloatingIslands()
    {
        int wakeBudget = wakeBudgetPerFrame;
        bool[] grounded = new bool[hashGridTotal];

        for (int x = 0; x < hashGridW; x++)
            grounded[x] = cellCounts[x] > 0;

        for (int y = 1; y < hashGridH; y++)
        for (int x = 0; x < hashGridW; x++)
        {
            int ci = y * hashGridW + x;
            if (cellCounts[ci] == 0) { grounded[ci] = false; continue; }

            bool sup = false;
            int yb = y - 1;
            if (x > 0) sup |= grounded[yb * hashGridW + (x - 1)];
            sup |= grounded[yb * hashGridW + x];
            if (x < hashGridW - 1) sup |= grounded[yb * hashGridW + (x + 1)];
            grounded[ci] = sup;
        }

        // Phase 2: Wake true islands
        for (int ci = 0; ci < hashGridTotal && wakeBudget > 0; ci++)
        {
            if (grounded[ci] || cellCounts[ci] == 0) continue;
            WakeSleepingInCell(ci, ref wakeBudget);
        }

        // Phase 3: Slope flow
        for (int y = 1; y < hashGridH && wakeBudget > 0; y++)
        for (int x = 0; x < hashGridW && wakeBudget > 0; x++)
        {
            int ci = y * hashGridW + x;
            if (!grounded[ci] || cellCounts[ci] == 0) continue;
            if (cellCounts[(y - 1) * hashGridW + x] > 0) continue;
            WakeSleepingInCell(ci, ref wakeBudget);
        }

        // Phase 4: Horizontal spreading
        for (int y = 1; y < hashGridH && wakeBudget > 0; y++)
        for (int x = 0; x < hashGridW && wakeBudget > 0; x++)
        {
            int ci = y * hashGridW + x;
            if (!grounded[ci] || cellCounts[ci] == 0) continue;

            bool flowLeft = false, flowRight = false;
            if (x > 0)
                flowLeft = cellCounts[y * hashGridW + (x - 1)] == 0
                        && cellCounts[(y - 1) * hashGridW + (x - 1)] == 0;
            if (x < hashGridW - 1)
                flowRight = cellCounts[y * hashGridW + (x + 1)] == 0
                         && cellCounts[(y - 1) * hashGridW + (x + 1)] == 0;

            if (flowLeft || flowRight) WakeSleepingInCell(ci, ref wakeBudget);
        }
    }

    void WakeSleepingInCell(int ci, ref int wakeBudget)
    {
        int start = cellOffsets[ci]; int count = cellCounts[ci];
        for (int s = 0; s < count && wakeBudget > 0; s++)
        {
            int i = sortedIndices[start + s];
            if (sleepState[i] != 1) continue;
            sleepState[i] = 0; sleepCounter[i] = 0; wakeBudget--;
        }
    }

    int2 CellCoord(float2 pos)
    {
        return math.clamp(
            (int2)math.floor((pos - new float2(containerMin.x, containerMin.y)) / cellSize),
            int2.zero, new int2(hashGridW - 1, hashGridH - 1));
    }

    // ═════════════════════════════════════════════════════════════
    //  GPU UPLOAD + DISPOSE
    // ═════════════════════════════════════════════════════════════

    void UploadToGPU()
    {
        int awake = 0;
        for (int i = 0; i < ParticleCount; i++)
        {
            var p = particles[i];
            Particles[i] = new FluidParticle
            {
                position = new Vector2(p.position.x, p.position.y),
                velocity = new Vector2(p.velocity.x, p.velocity.y),
                typeIndex = p.typeIndex,
                density = 0f, pressure = 0f,
                alive = p.alive,
                color = new Color(p.color.x, p.color.y, p.color.z, p.color.w)
            };
            if (p.alive > 0.5f && sleepState[i] == 0) awake++;
        }
        AwakeCount = awake;
        particleBuffer.SetData(Particles);
    }

    void DisposeNative()
    {
        if (particles.IsCreated) particles.Dispose();
        if (predictedPos.IsCreated) predictedPos.Dispose();
        if (sleepState.IsCreated) sleepState.Dispose();
        if (sleepCounter.IsCreated) sleepCounter.Dispose();
        if (bodyIndices.IsCreated) bodyIndices.Dispose();
        if (springs.IsCreated) springs.Dispose();
        if (cellCounts.IsCreated) cellCounts.Dispose();
        if (cellOffsets.IsCreated) cellOffsets.Dispose();
        if (sortedIndices.IsCreated) sortedIndices.Dispose();
        if (particleCellIndex.IsCreated) particleCellIndex.Dispose();
    }

    void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        Vector3 c = new Vector3((containerMin.x + containerMax.x) * 0.5f,
                                (containerMin.y + containerMax.y) * 0.5f, 0f);
        Vector3 s = new Vector3(containerMax.x - containerMin.x,
                                containerMax.y - containerMin.y, 0.01f);
        Gizmos.DrawWireCube(c, s);
    }

    // ═════════════════════════════════════════════════════════════
    //  JOBS — PBD PHYSICS
    // ═════════════════════════════════════════════════════════════

    // ─── Spatial hash: assign cells ──────────────────────────────

    [BurstCompile]
    struct AssignCellsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        public NativeArray<int> particleCellIndex;
        public float cellSize; public float2 containerMin;
        public int gridW, gridH;

        public void Execute(int i)
        {
            if (particles[i].alive < 0.5f) { particleCellIndex[i] = -1; return; }
            int2 cell = math.clamp(
                (int2)math.floor((particles[i].position - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1));
            particleCellIndex[i] = cell.y * gridW + cell.x;
        }
    }

    // ─── Spatial hash: build grid ────────────────────────────────

    [BurstCompile]
    struct BuildGridJob : IJob
    {
        [ReadOnly] public NativeArray<int> particleCellIndex;
        public NativeArray<int> cellCounts, cellOffsets, sortedIndices;
        public int particleCount, gridTotal;

        public void Execute()
        {
            for (int c = 0; c < gridTotal; c++) cellCounts[c] = 0;
            for (int i = 0; i < particleCount; i++)
                if (particleCellIndex[i] >= 0) cellCounts[particleCellIndex[i]]++;

            int off = 0;
            for (int c = 0; c < gridTotal; c++) { cellOffsets[c] = off; off += cellCounts[c]; }

            var tmp = new NativeArray<int>(gridTotal, Allocator.Temp);
            for (int c = 0; c < gridTotal; c++) tmp[c] = cellOffsets[c];
            for (int i = 0; i < particleCount; i++)
            {
                int ci = particleCellIndex[i];
                if (ci >= 0) sortedIndices[tmp[ci]++] = i;
            }
            tmp.Dispose();
        }
    }

    // ─── PBD Step 1: Predict positions ───────────────────────────

    [BurstCompile]
    struct PredictJob : IJobParallelFor
    {
        public NativeArray<ParticleData> particles;
        public NativeArray<float2> predictedPos;
        [ReadOnly] public NativeArray<int> sleepState;
        public float2 gravity; public float dt;

        public void Execute(int i)
        {
            var p = particles[i];

            // Sleeping or dead: predicted = current (no movement)
            if (p.alive < 0.5f || sleepState[i] == 1)
            {
                predictedPos[i] = p.position;
                return;
            }

            // Apply gravity to velocity, then predict
            p.velocity += gravity * dt;
            predictedPos[i] = p.position + p.velocity * dt;

            particles[i] = p;
        }
    }

    // ─── PBD Step 2: Spring constraint solver ────────────────────

    [BurstCompile]
    struct SpringSolveJob : IJob
    {
        public NativeArray<SpringData> springs;
        public NativeArray<float2> predictedPos;
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        public int springCount; public float stiffness;

        public void Execute()
        {
            for (int s = 0; s < springCount; s++)
            {
                var spring = springs[s];
                if (spring.alive == 0) continue;

                int a = spring.particleA;
                int b = spring.particleB;

                // Skip if both dead
                if (particles[a].alive < 0.5f || particles[b].alive < 0.5f) continue;

                bool aAwake = sleepState[a] == 0;
                bool bAwake = sleepState[b] == 0;

                // Skip if both sleeping
                if (!aAwake && !bAwake) continue;

                float2 posA = predictedPos[a];
                float2 posB = predictedPos[b];

                float2 delta = posB - posA;
                float dist = math.length(delta);

                if (dist < 1e-8f) continue;

                float diff = (dist - spring.restLength) / dist;
                float2 correction = delta * diff * stiffness * 0.5f;

                // If one is sleeping, it acts as an anchor — only the awake one moves
                if (aAwake && bAwake)
                {
                    predictedPos[a] = posA + correction;
                    predictedPos[b] = posB - correction;
                }
                else if (aAwake)
                {
                    // B is anchor — A gets full correction
                    predictedPos[a] = posA + correction * 2f;
                }
                else
                {
                    // A is anchor — B gets full correction
                    predictedPos[b] = posB - correction * 2f;
                }
            }
        }
    }

    // ─── PBD Step 3: Universal particle collision ─────────────────

    [BurstCompile]
    struct CollisionJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [NativeDisableParallelForRestriction]
        public NativeArray<float2> predictedPos;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<int> sortedIndices;
        [ReadOnly] public NativeArray<int> cellCounts, cellOffsets;
        public float cellSize; public float2 containerMin;
        public int gridW, gridH;
        public float collisionRadius, collisionStiffness;

        public void Execute(int i)
        {
            if (particles[i].alive < 0.5f || sleepState[i] == 1) return;

            float2 posI = predictedPos[i];
            float collRadSqr = collisionRadius * collisionRadius;

            int2 cellI = math.clamp(
                (int2)math.floor((posI - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1));

            float2 totalPush = float2.zero;
            int pushCount = 0;

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                int2 nc = cellI + new int2(dx, dy);
                if (nc.x < 0 || nc.x >= gridW || nc.y < 0 || nc.y >= gridH) continue;

                int ci = nc.y * gridW + nc.x;
                int start = cellOffsets[ci]; int count = cellCounts[ci];

                for (int s = 0; s < count; s++)
                {
                    int j = sortedIndices[start + s];
                    if (j == i) continue;
                    if (particles[j].alive < 0.5f) continue;

                    // Collide ALL particles — no body filter.
                    // Springs pull same-body particles together,
                    // collisions prevent overlap. They converge together.
                    float2 diff = posI - predictedPos[j];
                    float distSqr = math.lengthsq(diff);

                    if (distSqr < collRadSqr && distSqr > 1e-12f)
                    {
                        float dist = math.sqrt(distSqr);
                        float overlap = collisionRadius - dist;
                        float2 dir = diff / dist;
                        totalPush += dir * overlap * collisionStiffness;
                        pushCount++;
                    }
                }
            }

            if (pushCount > 0)
                predictedPos[i] = posI + totalPush / pushCount;
        }
    }

    // ─── PBD Step 4: Finalize positions + flask suction ──────────

    [BurstCompile]
    struct FinalizeJob : IJobParallelFor
    {
        public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<float2> predictedPos;
        [ReadOnly] public NativeArray<int> sleepState;

        public float dt, velocityDamping, maxSpeed;
        public float particleRadius, boundaryDamping;
        public float2 containerMin, containerMax;
        public bool flaskActive; public float2 flaskPos;
        public int flaskTargetType;
        public float flaskRadius, flaskAbsorbRadius, flaskStrength;

        public void Execute(int i)
        {
            var p = particles[i];
            if (p.alive < 0.5f || sleepState[i] == 1) return;

            // PBD velocity update: derive velocity from position change
            float2 newPos = predictedPos[i];
            p.velocity = (newPos - p.position) / math.max(dt, 1e-8f);

            // Flask suction (applied to velocity before finalization)
            if (flaskActive)
            {
                bool typeMatch = (flaskTargetType < 0) || (p.typeIndex == flaskTargetType);
                if (typeMatch)
                {
                    float2 toFlask = flaskPos - newPos;
                    float dist = math.length(toFlask);
                    if (dist < flaskRadius && dist > 0.001f)
                    {
                        float2 dir = toFlask / dist;
                        float pull = 1f - dist / flaskRadius;
                        p.velocity += dir * flaskStrength * pull * pull * dt;

                        if (dist < flaskAbsorbRadius)
                        {
                            p.alive = 0f;
                            p.position = new float2(-9999, -9999);
                            p.velocity = float2.zero;
                            particles[i] = p;
                            return;
                        }
                    }
                }
            }

            // Damping
            p.velocity *= velocityDamping;

            // Speed clamp
            float speedSqr = math.lengthsq(p.velocity);
            if (speedSqr > maxSpeed * maxSpeed)
                p.velocity *= maxSpeed / math.sqrt(speedSqr);

            // Final position
            p.position = newPos;

            // Boundaries
            float r = particleRadius;
            if (p.position.x < containerMin.x + r) { p.position.x = containerMin.x + r; p.velocity.x *= -boundaryDamping; }
            else if (p.position.x > containerMax.x - r) { p.position.x = containerMax.x - r; p.velocity.x *= -boundaryDamping; }
            if (p.position.y < containerMin.y + r) { p.position.y = containerMin.y + r; p.velocity.y *= -boundaryDamping; }
            else if (p.position.y > containerMax.y - r) { p.position.y = containerMax.y - r; p.velocity.y *= -boundaryDamping; }

            particles[i] = p;
        }
    }

    // ─── Spring break check ──────────────────────────────────────

    [BurstCompile]
    struct SpringBreakJob : IJob
    {
        public NativeArray<SpringData> springs;
        [ReadOnly] public NativeArray<ParticleData> particles;
        public int springCount;

        public void Execute()
        {
            for (int s = 0; s < springCount; s++)
            {
                var spring = springs[s];
                if (spring.alive == 0) continue;

                // Break if either particle is dead
                if (particles[spring.particleA].alive < 0.5f ||
                    particles[spring.particleB].alive < 0.5f)
                {
                    spring.alive = 0;
                    springs[s] = spring;
                    continue;
                }

                // Break if stretched beyond threshold
                float dist = math.length(
                    particles[spring.particleA].position -
                    particles[spring.particleB].position
                );

                if (dist > spring.restLength * spring.breakThreshold)
                {
                    spring.alive = 0;
                    springs[s] = spring;
                }
            }
        }
    }

    // ─── Sleep check ─────────────────────────────────────────────

    [BurstCompile]
    struct SleepCheckJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        public NativeArray<int> sleepState;
        public NativeArray<int> sleepCounter;
        public float sleepVelThresholdSqr;
        public int sleepFramesRequired;

        public void Execute(int i)
        {
            if (sleepState[i] == 1 || particles[i].alive < 0.5f) return;

            if (math.lengthsq(particles[i].velocity) < sleepVelThresholdSqr)
            {
                sleepCounter[i]++;
                if (sleepCounter[i] >= sleepFramesRequired)
                    sleepState[i] = 1;
            }
            else
            {
                sleepCounter[i] = 0;
            }
        }
    }
}