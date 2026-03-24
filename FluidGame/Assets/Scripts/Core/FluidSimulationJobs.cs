using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;

/// <summary>
/// CPU-based SPH fluid simulation with Jobs + Burst and a sleep/wake system.
///
/// SLEEP SYSTEM: Particles that aren't moving go to sleep and are excluded from
/// expensive SPH computations. Only particles near the flask or recently disturbed
/// areas are awake. This gives 10-15x speedup when most of the image is static.
///
/// WAKE BUDGET: Max N particles can wake per frame, preventing FPS spikes when
/// large areas collapse. Visually this creates a natural "cascading" effect.
/// </summary>
public class FluidSimulationJobs : MonoBehaviour
{
    // ─── Container ───────────────────────────────────────────────
    [Header("Container Bounds")]
    public Vector2 containerMin = new Vector2(-4f, -3f);
    public Vector2 containerMax = new Vector2(4f, 4f);

    // ─── Particle Spawning ───────────────────────────────────────
    [Header("Particle Grid")]
    public int gridWidth = 30;
    public int gridHeight = 20;
    public float particleSpacing = 0.15f;
    public float particleRadius = 0.05f;

    // ─── SPH Parameters ─────────────────────────────────────────
    [Header("SPH Settings")]
    public float smoothingRadius = 0.4f;
    public float particleMass = 1f;
    public float restDensity = 50f;
    public bool autoRestDensity = true;
    public float pressureStiffness = 80f;
    public float nearPressureStiffness = 5f;

    // ─── Cohesion & Separation ───────────────────────────────────
    [Header("Cohesion & Separation")]
    [Range(0f, 50f)]
    public float cohesionStrength = 15f;
    [Range(0f, 30f)]
    public float interTypeRepulsion = 8f;
    [Range(0f, 30f)]
    public float surfaceTensionStrength = 5f;
    public bool uniformFluid = false;

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
    [Tooltip("Particles with speed below this for sleepFrames will go to sleep")]
    public float sleepVelocityThreshold = 0.05f;

    [Tooltip("Consecutive low-velocity frames before sleeping")]
    public int sleepFramesRequired = 30;

    [Tooltip("Max particles that can wake up per FixedUpdate (prevents FPS spikes)")]
    public int wakeBudgetPerFrame = 150;

    [Tooltip("Radius around disturbance point to wake sleeping particles")]
    public float wakeRadius = 1.5f;

    [Tooltip("Start with all particles sleeping (for image mode — nothing moves until interacted with)")]
    public bool startSleeping = true;

    // ─── Fluid Types ─────────────────────────────────────────────
    [Header("Fluid Types")]
    public FluidTypeDefinition[] fluidTypes = new FluidTypeDefinition[]
    {
        new FluidTypeDefinition { name = "Heavy (Red)", color = new Color(0.9f, 0.2f, 0.15f),
            density = 3f, viscosity = 8f, cohesion = 1f },
        new FluidTypeDefinition { name = "Medium (Green)", color = new Color(0.2f, 0.85f, 0.3f),
            density = 2f, viscosity = 5f, cohesion = 0.8f },
        new FluidTypeDefinition { name = "Light (Blue)", color = new Color(0.2f, 0.4f, 0.95f),
            density = 1f, viscosity = 2f, cohesion = 0.6f },
    };

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

    // ─── Internal Native Data ────────────────────────────────────
    private NativeArray<ParticleData> particles;
    private NativeArray<float2> forces;
    private NativeArray<float> densities;
    private NativeArray<float> pressures;
    private NativeArray<FluidTypeGPU> fluidTypeData;

    // Sleep state (separate arrays to keep ParticleData at 48 bytes for GPU)
    private NativeArray<int> sleepState;     // 0 = awake, 1 = sleeping
    private NativeArray<int> sleepCounter;   // Frames at low velocity

    // Spatial hash arrays
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
    private struct ParticleData
    {
        public float2 position;
        public float2 velocity;
        public int typeIndex;
        public float density;
        public float pressure;
        public float alive;
        public float4 color;
    } // 48 bytes

    private struct FluidTypeGPU
    {
        public float gravityScale;
        public float viscosity;
        public float cohesion;
    }

    // ─── Lifecycle ───────────────────────────────────────────────

    void Awake()
    {
        var imageSource = GetComponent<ImageToFluid>();
        // ImageToFluid imageSource = null;
        if (imageSource != null)
        {
            imageSource.TryParseImage();
        }
        
        if (imageSource != null && imageSource.IsReady)
            InitFromImage(imageSource);
        else
            SpawnParticles();

        InitNativeData();
    }

    void FixedUpdate()
    {
        float dt = (Time.fixedDeltaTime * timeScale) / subSteps;

        // Wake particles near the flask every frame (with budget)
        if (flaskActive)
            WakeNearPoint(flaskPos, wakeRadius);

        for (int step = 0; step < subSteps; step++)
            RunSimulationStep(dt);

        // After simulation: check for void-waking (particles with no support below)
        WakeUnsupported();

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
                    velocity = Vector2.zero,
                    typeIndex = typeIndex,
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
        startSleeping = true; // Image particles start asleep

        float s = particleSpacing;
        smoothingRadius = s * 2.5f;
        particleRadius = s * 0.4f;
        nearPressureStiffness = 10f;
        pressureStiffness = 120f;
        subSteps = Mathf.Max(subSteps, 4);
        wakeRadius = smoothingRadius * 4f;

        Debug.Log($"[FluidSimJobs] From image: {ParticleCount} particles, " +
                  $"spacing={s:F4}, h={smoothingRadius:F4}");
    }

    void InitNativeData()
    {
        cellSize = smoothingRadius;
        float containerW = containerMax.x - containerMin.x;
        float containerH = containerMax.y - containerMin.y;
        hashGridW = Mathf.CeilToInt(containerW / cellSize) + 1;
        hashGridH = Mathf.CeilToInt(containerH / cellSize) + 1;
        hashGridTotal = hashGridW * hashGridH;

        particles = new NativeArray<ParticleData>(ParticleCount, Allocator.Persistent);
        forces = new NativeArray<float2>(ParticleCount, Allocator.Persistent);
        densities = new NativeArray<float>(ParticleCount, Allocator.Persistent);
        pressures = new NativeArray<float>(ParticleCount, Allocator.Persistent);
        sleepState = new NativeArray<int>(ParticleCount, Allocator.Persistent);
        sleepCounter = new NativeArray<int>(ParticleCount, Allocator.Persistent);

        cellCounts = new NativeArray<int>(hashGridTotal, Allocator.Persistent);
        cellOffsets = new NativeArray<int>(hashGridTotal, Allocator.Persistent);
        sortedIndices = new NativeArray<int>(ParticleCount, Allocator.Persistent);
        particleCellIndex = new NativeArray<int>(ParticleCount, Allocator.Persistent);

        // Build fluid type data
        float avgDensity = 0f;
        for (int i = 0; i < fluidTypes.Length; i++) avgDensity += fluidTypes[i].density;
        avgDensity /= fluidTypes.Length;

        fluidTypeData = new NativeArray<FluidTypeGPU>(fluidTypes.Length, Allocator.Persistent);
        for (int i = 0; i < fluidTypes.Length; i++)
        {
            fluidTypeData[i] = new FluidTypeGPU
            {
                gravityScale = fluidTypes[i].density / avgDensity,
                viscosity = fluidTypes[i].viscosity,
                cohesion = fluidTypes[i].cohesion
            };
        }

        // Copy managed → native, set initial sleep state
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

            // Start sleeping if configured (image mode) or awake (grid mode)
            sleepState[i] = (startSleeping && p.alive > 0.5f) ? 1 : 0;
            sleepCounter[i] = startSleeping ? sleepFramesRequired : 0;
        }

        if (autoRestDensity) CalibrateRestDensity();

        particleBuffer = new ComputeBuffer(ParticleCount, 48);
        UploadToGPU();

        AwakeCount = startSleeping ? 0 : ParticleCount;
        Debug.Log($"[FluidSimJobs] Init: {ParticleCount} particles, " +
                  $"grid {hashGridW}x{hashGridH}, rest={restDensity:F1}, " +
                  $"awake={AwakeCount}, sleep={startSleeping}");
    }

    void CalibrateRestDensity()
    {
        float h = smoothingRadius;
        float hSqr = h * h;
        float h8 = hSqr * hSqr * hSqr * hSqr;
        float coeff = 4f / (math.PI * h8);

        float2 minP = new float2(float.MaxValue), maxP = new float2(float.MinValue);
        for (int i = 0; i < ParticleCount; i++)
        {
            float2 pos = particles[i].position;
            minP = math.min(minP, pos);
            maxP = math.max(maxP, pos);
        }
        float2 center = (minP + maxP) * 0.5f;
        float sampleRadSqr = math.lengthsq((maxP - minP) * 0.25f);

        float totalDensity = 0f;
        int sampleCount = 0;

        for (int i = 0; i < ParticleCount && sampleCount < 200; i++)
        {
            if (math.lengthsq(particles[i].position - center) > sampleRadSqr) continue;
            float density = 0f;
            for (int j = 0; j < ParticleCount; j++)
            {
                float rSqr = math.lengthsq(particles[i].position - particles[j].position);
                if (rSqr < hSqr)
                {
                    float d = hSqr - rSqr;
                    density += particleMass * coeff * d * d * d;
                }
            }
            totalDensity += density;
            sampleCount++;
        }

        if (sampleCount > 0)
            restDensity = (totalDensity / sampleCount) * (uniformFluid ? 1f : 0.95f);

        Debug.Log($"[FluidSimJobs] restDensity = {restDensity:F1} ({sampleCount} samples)");
    }

    // ─── Wake / Sleep Logic (main thread) ────────────────────────

    private int wakesBudgetRemaining;

    /// <summary>
    /// Wakes sleeping particles within radius of a point, up to the per-frame budget.
    /// Called when flask is active (suction pulls particles, exposing void).
    /// </summary>
    void WakeNearPoint(float2 point, float radius)
    {
        wakesBudgetRemaining = wakeBudgetPerFrame;
        float radiusSqr = radius * radius;

        for (int i = 0; i < ParticleCount && wakesBudgetRemaining > 0; i++)
        {
            if (sleepState[i] == 0) continue; // Already awake
            if (particles[i].alive < 0.5f) continue; // Dead

            float distSqr = math.lengthsq(particles[i].position - point);
            if (distSqr < radiusSqr)
            {
                sleepState[i] = 0;
                sleepCounter[i] = 0;
                wakesBudgetRemaining--;
            }
        }
    }

    /// <summary>
    /// Wakes sleeping particles that have no support below them
    /// (no sleeping or awake neighbor within smoothingRadius below).
    /// This handles cascading collapse when lower particles are absorbed.
    /// Runs with a budget to prevent mass wake-up.
    /// </summary>
    void WakeUnsupported()
    {
        // Only check periodically to save CPU (every 5 frames)
        if (Time.frameCount % 5 != 0) return;

        int budget = wakeBudgetPerFrame;
        float checkBelow = smoothingRadius * 1.5f;

        for (int i = 0; i < ParticleCount && budget > 0; i++)
        {
            if (sleepState[i] == 0) continue; // Already awake
            if (particles[i].alive < 0.5f) continue;

            // Check: is there any alive particle below me within range?
            float2 pos = particles[i].position;
            bool hasSupport = false;

            // Quick check using the spatial hash
            int2 cell = CellCoord(pos);
            for (int dx = -1; dx <= 1 && !hasSupport; dx++)
            for (int dy = -1; dy <= 0 && !hasSupport; dy++) // Only check same row and below
            {
                int2 nc = cell + new int2(dx, dy);
                if (nc.x < 0 || nc.x >= hashGridW || nc.y < 0 || nc.y >= hashGridH) continue;

                int ci = nc.y * hashGridW + nc.x;
                int start = cellOffsets[ci];
                int count = cellCounts[ci];

                for (int s = 0; s < count; s++)
                {
                    int j = sortedIndices[start + s];
                    if (j == i) continue;
                    if (particles[j].alive < 0.5f) continue;

                    float2 diff = particles[j].position - pos;
                    // Neighbor must be below (or at same level) and within range
                    if (diff.y <= 0.01f && math.lengthsq(diff) < checkBelow * checkBelow)
                    {
                        hasSupport = true;
                    }
                }
            }

            // Also count bottom boundary as support
            if (pos.y < containerMin.y + particleRadius + smoothingRadius)
                hasSupport = true;

            if (!hasSupport)
            {
                sleepState[i] = 0;
                sleepCounter[i] = 0;
                budget--;
            }
        }
    }

    int2 CellCoord(float2 pos)
    {
        return math.clamp(
            (int2)math.floor((pos - new float2(containerMin.x, containerMin.y)) / cellSize),
            int2.zero, new int2(hashGridW - 1, hashGridH - 1));
    }

    // ─── Simulation Step ─────────────────────────────────────────

    void RunSimulationStep(float dt)
    {
        // 1. Build spatial hash — ALL alive particles (sleeping + awake)
        //    Sleeping particles must be in the grid so awake neighbors sense them for density.
        var assignJob = new AssignCellsJob
        {
            particles = particles,
            particleCellIndex = particleCellIndex,
            cellSize = cellSize,
            containerMin = new float2(containerMin.x, containerMin.y),
            gridW = hashGridW, gridH = hashGridH
        };
        assignJob.Schedule(ParticleCount, 128).Complete();

        var buildGridJob = new BuildGridJob
        {
            particleCellIndex = particleCellIndex,
            cellCounts = cellCounts,
            cellOffsets = cellOffsets,
            sortedIndices = sortedIndices,
            particleCount = ParticleCount,
            gridTotal = hashGridTotal
        };
        buildGridJob.Run();

        // 2. Density — only for awake particles, but reads ALL neighbors
        var densityJob = new DensityJob
        {
            particles = particles,
            sleepState = sleepState,
            sortedIndices = sortedIndices,
            cellCounts = cellCounts,
            cellOffsets = cellOffsets,
            densities = densities,
            pressures = pressures,
            cellSize = cellSize,
            containerMin = new float2(containerMin.x, containerMin.y),
            gridW = hashGridW, gridH = hashGridH,
            smoothingRadiusSqr = smoothingRadius * smoothingRadius,
            particleMass = particleMass,
            restDensity = restDensity,
            pressureStiffness = pressureStiffness,
            poly6Coeff = 4f / (math.PI * math.pow(smoothingRadius, 8))
        };
        densityJob.Schedule(ParticleCount, 128).Complete();

        // 3. Forces — only for awake particles
        var forcesJob = new ForcesJob
        {
            particles = particles,
            sleepState = sleepState,
            sortedIndices = sortedIndices,
            cellCounts = cellCounts,
            cellOffsets = cellOffsets,
            densities = densities,
            pressures = pressures,
            forces = forces,
            fluidTypeData = fluidTypeData,
            cellSize = cellSize,
            containerMin = new float2(containerMin.x, containerMin.y),
            gridW = hashGridW, gridH = hashGridH,
            smoothingRadius = smoothingRadius,
            smoothingRadiusSqr = smoothingRadius * smoothingRadius,
            particleMass = particleMass,
            nearPressureStiffness = nearPressureStiffness,
            cohesionStrength = cohesionStrength,
            interTypeRepulsion = interTypeRepulsion,
            surfaceTensionStrength = surfaceTensionStrength,
            uniformFluid = uniformFluid,
            spikyGradCoeff = -10f / (math.PI * math.pow(smoothingRadius, 5)),
            viscLaplCoeff = 40f / (math.PI * math.pow(smoothingRadius, 5))
        };
        forcesJob.Schedule(ParticleCount, 64).Complete();

        // 4. Integrate — only awake particles, includes suction
        var integrateJob = new IntegrateJob
        {
            particles = particles,
            sleepState = sleepState,
            forces = forces,
            densities = densities,
            fluidTypeData = fluidTypeData,
            gravity = new float2(gravity.x, gravity.y),
            dt = dt,
            velocityDamping = velocityDamping,
            maxSpeed = maxSpeed,
            particleRadius = particleRadius,
            boundaryDamping = boundaryDamping,
            containerMin = new float2(containerMin.x, containerMin.y),
            containerMax = new float2(containerMax.x, containerMax.y),
            uniformFluid = uniformFluid,
            flaskActive = flaskActive,
            flaskPos = flaskPos,
            flaskTargetType = flaskTargetType,
            flaskRadius = flaskRadius,
            flaskAbsorbRadius = flaskAbsorbRadius,
            flaskStrength = flaskStrength
        };
        integrateJob.Schedule(ParticleCount, 128).Complete();

        // 5. Sleep check — awake particles with low velocity go to sleep
        var sleepJob = new SleepCheckJob
        {
            particles = particles,
            sleepState = sleepState,
            sleepCounter = sleepCounter,
            sleepVelThresholdSqr = sleepVelocityThreshold * sleepVelocityThreshold,
            sleepFramesRequired = sleepFramesRequired
        };
        sleepJob.Schedule(ParticleCount, 256).Complete();
    }

    // ─── GPU Upload ──────────────────────────────────────────────

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
                density = densities[i],
                pressure = pressures[i],
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
        if (forces.IsCreated) forces.Dispose();
        if (densities.IsCreated) densities.Dispose();
        if (pressures.IsCreated) pressures.Dispose();
        if (sleepState.IsCreated) sleepState.Dispose();
        if (sleepCounter.IsCreated) sleepCounter.Dispose();
        if (cellCounts.IsCreated) cellCounts.Dispose();
        if (cellOffsets.IsCreated) cellOffsets.Dispose();
        if (sortedIndices.IsCreated) sortedIndices.Dispose();
        if (particleCellIndex.IsCreated) particleCellIndex.Dispose();
        if (fluidTypeData.IsCreated) fluidTypeData.Dispose();
    }

    void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        Vector3 center = new Vector3((containerMin.x + containerMax.x) * 0.5f,
                                     (containerMin.y + containerMax.y) * 0.5f, 0f);
        Vector3 size = new Vector3(containerMax.x - containerMin.x,
                                   containerMax.y - containerMin.y, 0.01f);
        Gizmos.DrawWireCube(center, size);
    }

    // ═════════════════════════════════════════════════════════════
    //  JOBS
    // ═════════════════════════════════════════════════════════════

    // ─── Assign cell index (parallel) ────────────────────────────

    [BurstCompile]
    struct AssignCellsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        public NativeArray<int> particleCellIndex;
        public float cellSize;
        public float2 containerMin;
        public int gridW, gridH;

        public void Execute(int i)
        {
            // ALL alive particles go in the grid (sleeping + awake)
            if (particles[i].alive < 0.5f)
            {
                particleCellIndex[i] = -1;
                return;
            }
            int2 cell = math.clamp(
                (int2)math.floor((particles[i].position - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1));
            particleCellIndex[i] = cell.y * gridW + cell.x;
        }
    }

    // ─── Build grid (single-threaded, Burst) ─────────────────────

    [BurstCompile]
    struct BuildGridJob : IJob
    {
        [ReadOnly] public NativeArray<int> particleCellIndex;
        public NativeArray<int> cellCounts;
        public NativeArray<int> cellOffsets;
        public NativeArray<int> sortedIndices;
        public int particleCount, gridTotal;

        public void Execute()
        {
            for (int c = 0; c < gridTotal; c++) cellCounts[c] = 0;
            for (int i = 0; i < particleCount; i++)
            {
                int ci = particleCellIndex[i];
                if (ci >= 0) cellCounts[ci]++;
            }

            int offset = 0;
            for (int c = 0; c < gridTotal; c++)
            {
                cellOffsets[c] = offset;
                offset += cellCounts[c];
            }

            var tempOff = new NativeArray<int>(gridTotal, Allocator.Temp);
            for (int c = 0; c < gridTotal; c++) tempOff[c] = cellOffsets[c];
            for (int i = 0; i < particleCount; i++)
            {
                int ci = particleCellIndex[i];
                if (ci < 0) continue;
                sortedIndices[tempOff[ci]++] = i;
            }
            tempOff.Dispose();
        }
    }

    // ─── Density (parallel, skips sleeping) ──────────────────────

    [BurstCompile]
    struct DensityJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<int> sortedIndices;
        [ReadOnly] public NativeArray<int> cellCounts;
        [ReadOnly] public NativeArray<int> cellOffsets;
        [NativeDisableParallelForRestriction] public NativeArray<float> densities;
        [NativeDisableParallelForRestriction] public NativeArray<float> pressures;

        public float cellSize, smoothingRadiusSqr;
        public float2 containerMin;
        public int gridW, gridH;
        public float particleMass, restDensity, pressureStiffness, poly6Coeff;

        public void Execute(int i)
        {
            // SKIP sleeping particles — they don't need updated density
            if (sleepState[i] == 1 || particles[i].alive < 0.5f)
            {
                densities[i] = restDensity; // Stable placeholder
                pressures[i] = 0f;
                return;
            }

            float2 posI = particles[i].position;
            int2 cellI = math.clamp(
                (int2)math.floor((posI - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1));

            float density = 0f;

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                int2 nc = cellI + new int2(dx, dy);
                if (nc.x < 0 || nc.x >= gridW || nc.y < 0 || nc.y >= gridH) continue;

                int ci = nc.y * gridW + nc.x;
                int start = cellOffsets[ci];
                int count = cellCounts[ci];

                // Reads ALL neighbors (sleeping + awake) so density is correct
                for (int s = 0; s < count; s++)
                {
                    int j = sortedIndices[start + s];
                    float rSqr = math.lengthsq(posI - particles[j].position);
                    if (rSqr < smoothingRadiusSqr)
                    {
                        float diff = smoothingRadiusSqr - rSqr;
                        density += particleMass * poly6Coeff * diff * diff * diff;
                    }
                }
            }

            density = math.max(density, 0.001f);
            densities[i] = density;
            pressures[i] = math.max(0f, pressureStiffness * (density - restDensity));
        }
    }

    // ─── Forces (parallel, skips sleeping) ───────────────────────

    [BurstCompile]
    struct ForcesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<int> sortedIndices;
        [ReadOnly] public NativeArray<int> cellCounts;
        [ReadOnly] public NativeArray<int> cellOffsets;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<float> pressures;
        [NativeDisableParallelForRestriction] public NativeArray<float2> forces;
        [ReadOnly] public NativeArray<FluidTypeGPU> fluidTypeData;

        public float cellSize, smoothingRadius, smoothingRadiusSqr, particleMass;
        public float2 containerMin;
        public int gridW, gridH;
        public float nearPressureStiffness;
        public float cohesionStrength, interTypeRepulsion, surfaceTensionStrength;
        public bool uniformFluid;
        public float spikyGradCoeff, viscLaplCoeff;

        public void Execute(int i)
        {
            // SKIP sleeping and dead particles
            if (sleepState[i] == 1 || particles[i].alive < 0.5f)
            {
                forces[i] = float2.zero;
                return;
            }

            var pI = particles[i];
            int2 cellI = math.clamp(
                (int2)math.floor((pI.position - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1));
            var typeI = fluidTypeData[pI.typeIndex];

            float2 totalForce = float2.zero;
            float2 sameTypeCOM = float2.zero;
            float sameTypeWeight = 0f;

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                int2 nc = cellI + new int2(dx, dy);
                if (nc.x < 0 || nc.x >= gridW || nc.y < 0 || nc.y >= gridH) continue;

                int ci = nc.y * gridW + nc.x;
                int start = cellOffsets[ci];
                int count = cellCounts[ci];

                for (int s = 0; s < count; s++)
                {
                    int j = sortedIndices[start + s];
                    if (j == i) continue;

                    var pJ = particles[j];
                    if (pJ.alive < 0.5f) continue;

                    float2 diff = pI.position - pJ.position;
                    float rSqr = math.lengthsq(diff);
                    if (rSqr >= smoothingRadiusSqr || rSqr < 1e-12f) continue;

                    float r = math.sqrt(rSqr);
                    float2 dir = diff / r;

                    float densityJ = densities[j];
                    // For sleeping neighbors, use restDensity as stable estimate
                    if (densityJ < 0.01f) densityJ = math.max(densities[j], 0.001f);

                    // Pressure
                    float gm = SpikyGrad(r);
                    float pressAvg = (pressures[i] + pressures[j]) * 0.5f;
                    float2 pF = dir * (-particleMass * pressAvg * gm / math.max(densityJ, 0.001f));

                    // Near-pressure
                    float nf = 1f - r / smoothingRadius;
                    float2 nF = dir * (nearPressureStiffness * nf * nf);

                    // Viscosity
                    var typeJ = fluidTypeData[pJ.typeIndex];
                    float mu = (typeI.viscosity + typeJ.viscosity) * 0.5f;
                    float vl = ViscLapl(r);
                    float2 vF = mu * particleMass * (pJ.velocity - pI.velocity)
                              / math.max(densityJ, 0.001f) * vl;

                    bool same = uniformFluid || (pI.typeIndex == pJ.typeIndex);

                    float2 cF = float2.zero;
                    if (same)
                    {
                        float t = r / smoothingRadius;
                        cF = -dir * typeI.cohesion * cohesionStrength * t * (1f - t) * (1f - t);
                        float w = 1f - t;
                        sameTypeCOM += pJ.position * w;
                        sameTypeWeight += w;
                    }

                    float2 rF = float2.zero;
                    if (!same)
                    {
                        float rf = 1f - r / smoothingRadius;
                        rF = dir * interTypeRepulsion * rf * rf;
                    }

                    totalForce += pF + nF + vF + cF + rF;
                }
            }

            if (sameTypeWeight > 0.001f)
            {
                float2 com = sameTypeCOM / sameTypeWeight;
                float2 toCOM = com - pI.position;
                float d = math.length(toCOM);
                if (d > 0.001f)
                    totalForce += (toCOM / d) * surfaceTensionStrength * typeI.cohesion
                                * math.min(d, smoothingRadius);
            }

            forces[i] = totalForce;
        }

        float SpikyGrad(float r)
        {
            if (r >= smoothingRadius || r < 1e-6f) return 0f;
            float d = smoothingRadius - r;
            return spikyGradCoeff * d * d;
        }

        float ViscLapl(float r)
        {
            if (r >= smoothingRadius || r < 1e-6f) return 0f;
            return viscLaplCoeff * (smoothingRadius - r);
        }
    }

    // ─── Integrate (parallel, skips sleeping) ────────────────────

    [BurstCompile]
    struct IntegrateJob : IJobParallelFor
    {
        public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<float2> forces;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<FluidTypeGPU> fluidTypeData;

        public float2 gravity, containerMin, containerMax;
        public float dt, velocityDamping, maxSpeed, particleRadius, boundaryDamping;
        public bool uniformFluid, flaskActive;
        public float2 flaskPos;
        public int flaskTargetType;
        public float flaskRadius, flaskAbsorbRadius, flaskStrength;

        public void Execute(int i)
        {
            var p = particles[i];
            if (p.alive < 0.5f || sleepState[i] == 1) return; // Skip dead and sleeping

            var ft = fluidTypeData[p.typeIndex];
            float gScale = uniformFluid ? 1f : ft.gravityScale;
            float density = math.max(densities[i], 0.001f);
            float2 accel = forces[i] / density + gravity * gScale;

            p.velocity += accel * dt;

            // Flask suction
            if (flaskActive)
            {
                bool typeMatch = (flaskTargetType < 0) || (p.typeIndex == flaskTargetType);
                if (typeMatch)
                {
                    float2 toFlask = flaskPos - p.position;
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

            p.velocity *= velocityDamping;

            float speedSqr = math.lengthsq(p.velocity);
            if (speedSqr > maxSpeed * maxSpeed)
                p.velocity *= maxSpeed / math.sqrt(speedSqr);

            p.position += p.velocity * dt;

            // Boundaries
            float r = particleRadius;
            if (p.position.x < containerMin.x + r) { p.position.x = containerMin.x + r; p.velocity.x *= -boundaryDamping; }
            else if (p.position.x > containerMax.x - r) { p.position.x = containerMax.x - r; p.velocity.x *= -boundaryDamping; }
            if (p.position.y < containerMin.y + r) { p.position.y = containerMin.y + r; p.velocity.y *= -boundaryDamping; }
            else if (p.position.y > containerMax.y - r) { p.position.y = containerMax.y - r; p.velocity.y *= -boundaryDamping; }

            particles[i] = p;
        }
    }

    // ─── Sleep check (parallel) ──────────────────────────────────

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
            // Only check awake, alive particles
            if (sleepState[i] == 1 || particles[i].alive < 0.5f) return;

            float speedSqr = math.lengthsq(particles[i].velocity);

            if (speedSqr < sleepVelThresholdSqr)
            {
                sleepCounter[i]++;
                if (sleepCounter[i] >= sleepFramesRequired)
                {
                    sleepState[i] = 1; // Go to sleep
                }
            }
            else
            {
                sleepCounter[i] = 0; // Reset counter — still moving
            }
        }
    }
}