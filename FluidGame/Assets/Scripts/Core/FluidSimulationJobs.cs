using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;

/// <summary>
/// CPU-based SPH fluid simulation using Unity Jobs + Burst compiler.
/// Drop-in alternative to FluidSimulationGPU — same public API,
/// same particle struct layout, same ComputeBuffer for rendering.
///
/// SETUP: Add this instead of (or alongside) FluidSimulationGPU.
///        Disable whichever one you don't want active.
///        Renderers (MetaballFluidRenderer, FluidRendererGPU) work with both.
///
/// REQUIRES: com.unity.burst, com.unity.collections packages.
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

    // ─── Public Accessors (same API as FluidSimulationGPU) ───────
    public FluidParticle[] Particles { get; private set; }
    public int ParticleCount { get; private set; }
    public ComputeBuffer ParticleBuffer => particleBuffer;

    // ─── Internal Native Data ────────────────────────────────────
    private NativeArray<ParticleData> particles;
    private NativeArray<float2> forces;
    private NativeArray<float> densities;
    private NativeArray<float> pressures;
    private NativeArray<FluidTypeGPU> fluidTypeData;

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

    // ─── Burst-compatible particle struct ────────────────────────
    // Must produce the same 48-byte layout when copied to FluidParticle[]
    private struct ParticleData
    {
        public float2 position;     // 8
        public float2 velocity;     // 8
        public int typeIndex;       // 4
        public float density;       // 4
        public float pressure;      // 4
        public float alive;         // 4
        public float4 color;        // 16
    }                               // Total: 48

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

        for (int step = 0; step < subSteps; step++)
            RunSimulationStep(dt);

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

        Debug.Log($"[FluidSimJobs] Spawned {ParticleCount} particles (grid {gridWidth}x{gridHeight})");
    }

    void InitFromImage(ImageToFluid source)
    {
        Particles = source.GeneratedParticles;
        ParticleCount = source.GeneratedParticleCount;
        fluidTypes = source.GeneratedFluidTypes;
        particleSpacing = source.ComputedSpacing;
        uniformFluid = true;

        float s = particleSpacing;
        smoothingRadius = s * 2.5f;
        particleRadius = s * 0.4f;
        nearPressureStiffness = 10f;
        pressureStiffness = 120f;
        subSteps = Mathf.Max(subSteps, 4);

        Debug.Log($"[FluidSimJobs] Initialized from image: {ParticleCount} particles, " +
                  $"spacing={s:F4}, smoothingRadius={smoothingRadius:F4}");
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
        forces = new NativeArray<float2>(ParticleCount, Allocator.Persistent);
        densities = new NativeArray<float>(ParticleCount, Allocator.Persistent);
        pressures = new NativeArray<float>(ParticleCount, Allocator.Persistent);

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

        // Copy managed Particles → native
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
        }

        // Auto-calibrate rest density
        if (autoRestDensity)
            CalibrateRestDensity();

        // Create GPU buffer for renderer bridge
        particleBuffer = new ComputeBuffer(ParticleCount, 48);
        UploadToGPU();

        Debug.Log($"[FluidSimJobs] Initialized: {ParticleCount} particles, " +
                  $"grid {hashGridW}x{hashGridH}, restDensity={restDensity:F1}");
    }

    void CalibrateRestDensity()
    {
        float h = smoothingRadius;
        float hSqr = h * h;
        float h8 = hSqr * hSqr * hSqr * hSqr;
        float coeff = 4f / (math.PI * h8);

        // Find center of particle cloud
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
        {
            float mult = uniformFluid ? 1f : 0.95f;
            restDensity = (totalDensity / sampleCount) * mult;
        }

        Debug.Log($"[FluidSimJobs] Calibrated restDensity = {restDensity:F1} ({sampleCount} samples)");
    }

    // ─── Simulation Step ─────────────────────────────────────────

    void RunSimulationStep(float dt)
    {
        // 1. Build spatial hash (counting sort — no atomics, no races)

        // Phase A: Assign cell indices (parallel — safe, each writes own index)
        var assignJob = new AssignCellsJob
        {
            particles = particles,
            particleCellIndex = particleCellIndex,
            cellSize = cellSize,
            containerMin = new float2(containerMin.x, containerMin.y),
            gridW = hashGridW, gridH = hashGridH
        };
        assignJob.Schedule(ParticleCount, 128).Complete();

        // Phase B: Count + prefix sum + scatter (single-threaded — fast with Burst)
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

        // 2. Compute density and pressure for all particles (must complete before forces)
        var densityJob = new DensityJob
        {
            particles = particles,
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

        // 3. Compute all forces (reads density/pressure computed above)
        var sphJob = new SPHForcesJob
        {
            particles = particles,
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
        sphJob.Schedule(ParticleCount, 64).Complete();

        // 4. Integration
        var integrateJob = new IntegrateJob
        {
            particles = particles,
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
    }

    // (prefix sum is now inside BuildGridJob)

    // ─── GPU Upload ──────────────────────────────────────────────

    /// <summary>
    /// Copies native particle data into the ComputeBuffer so GPU renderers work.
    /// Also syncs back to managed Particles[] for FlaskUI/debug.
    /// </summary>
    void UploadToGPU()
    {
        // Native → managed (for UI/debug readback)
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
        }

        // Managed → GPU buffer (for shaders)
        particleBuffer.SetData(Particles);
    }

    void DisposeNative()
    {
        if (particles.IsCreated) particles.Dispose();
        if (forces.IsCreated) forces.Dispose();
        if (densities.IsCreated) densities.Dispose();
        if (pressures.IsCreated) pressures.Dispose();
        if (cellCounts.IsCreated) cellCounts.Dispose();
        if (cellOffsets.IsCreated) cellOffsets.Dispose();
        if (sortedIndices.IsCreated) sortedIndices.Dispose();
        if (particleCellIndex.IsCreated) particleCellIndex.Dispose();
        if (fluidTypeData.IsCreated) fluidTypeData.Dispose();
    }

    // ─── Debug ───────────────────────────────────────────────────

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

    // ─── Job 1a: Assign cell index per particle (parallel) ─────

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
            if (particles[i].alive < 0.5f)
            {
                particleCellIndex[i] = -1;
                return;
            }

            int2 cell = math.clamp(
                (int2)math.floor((particles[i].position - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1)
            );
            particleCellIndex[i] = cell.y * gridW + cell.x;
        }
    }

    // ─── Job 1b: Count, prefix sum, scatter (single-threaded) ────

    [BurstCompile]
    struct BuildGridJob : IJob
    {
        [ReadOnly] public NativeArray<int> particleCellIndex;
        public NativeArray<int> cellCounts;
        public NativeArray<int> cellOffsets;
        public NativeArray<int> sortedIndices;
        public int particleCount;
        public int gridTotal;

        public void Execute()
        {
            // Clear
            for (int c = 0; c < gridTotal; c++)
                cellCounts[c] = 0;

            // Count
            for (int i = 0; i < particleCount; i++)
            {
                int ci = particleCellIndex[i];
                if (ci >= 0) cellCounts[ci]++;
            }

            // Prefix sum
            int offset = 0;
            for (int c = 0; c < gridTotal; c++)
            {
                cellOffsets[c] = offset;
                offset += cellCounts[c];
            }

            // Scatter (using temp copy of offsets)
            var tempOffsets = new NativeArray<int>(gridTotal, Allocator.Temp);
            for (int c = 0; c < gridTotal; c++)
                tempOffsets[c] = cellOffsets[c];

            for (int i = 0; i < particleCount; i++)
            {
                int ci = particleCellIndex[i];
                if (ci < 0) continue;
                sortedIndices[tempOffsets[ci]++] = i;
            }

            tempOffsets.Dispose();
        }
    }

    // ─── Job 2: Density + Pressure ─────────────────────────────────

    [BurstCompile]
    struct DensityJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sortedIndices;
        [ReadOnly] public NativeArray<int> cellCounts;
        [ReadOnly] public NativeArray<int> cellOffsets;

        // Output: density and pressure written to separate arrays (not back to particles)
        // This avoids the read-write race that Unity Safety System catches.
        [NativeDisableParallelForRestriction]
        public NativeArray<float> densities;
        [NativeDisableParallelForRestriction]
        public NativeArray<float> pressures;

        public float cellSize;
        public float2 containerMin;
        public int gridW, gridH;
        public float smoothingRadiusSqr;
        public float particleMass, restDensity, pressureStiffness;
        public float poly6Coeff;

        public void Execute(int i)
        {
            if (particles[i].alive < 0.5f)
            {
                densities[i] = 0f;
                pressures[i] = 0f;
                return;
            }

            float2 posI = particles[i].position;
            int2 cellI = CellCoord(posI);
            float density = 0f;

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

        int2 CellCoord(float2 pos)
        {
            return math.clamp(
                (int2)math.floor((pos - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1));
        }
    }

    // ─── Job 3: Forces (reads density/pressure written by DensityJob) ─

    [BurstCompile]
    struct SPHForcesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sortedIndices;
        [ReadOnly] public NativeArray<int> cellCounts;
        [ReadOnly] public NativeArray<int> cellOffsets;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<float> pressures;
        [NativeDisableParallelForRestriction]
        public NativeArray<float2> forces;
        [ReadOnly] public NativeArray<FluidTypeGPU> fluidTypeData;

        public float cellSize;
        public float2 containerMin;
        public int gridW, gridH;
        public float smoothingRadius, smoothingRadiusSqr;
        public float particleMass;
        public float nearPressureStiffness;
        public float cohesionStrength, interTypeRepulsion, surfaceTensionStrength;
        public bool uniformFluid;
        public float spikyGradCoeff, viscLaplCoeff;

        public void Execute(int i)
        {
            var pI = particles[i];
            if (pI.alive < 0.5f)
            {
                forces[i] = float2.zero;
                return;
            }

            int2 cellI = CellCoord(pI.position);
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
                    var typeJ = fluidTypeData[pJ.typeIndex];

                    // Pressure (Spiky gradient)
                    float gradMag = SpikyGrad(r);
                    float densityJ = densities[j];
                    float pressAvg = (pressures[i] + pressures[j]) * 0.5f;
                    float2 pressForce = dir * (-particleMass * pressAvg * gradMag / densityJ);

                    // Near-pressure
                    float nearFactor = 1f - r / smoothingRadius;
                    float2 nearForce = dir * (nearPressureStiffness * nearFactor * nearFactor);

                    // Viscosity
                    float mu = (typeI.viscosity + typeJ.viscosity) * 0.5f;
                    float viscLapl = ViscLaplacian(r);
                    float2 viscForce = mu * particleMass * (pJ.velocity - pI.velocity)
                                     / math.max(densityJ, 0.001f) * viscLapl;

                    bool sameType = uniformFluid || (pI.typeIndex == pJ.typeIndex);

                    // Cohesion
                    float2 cohesionForce = float2.zero;
                    if (sameType)
                    {
                        float t = r / smoothingRadius;
                        float attraction = t * (1f - t) * (1f - t);
                        cohesionForce = -dir * typeI.cohesion * cohesionStrength * attraction;

                        float w = 1f - t;
                        sameTypeCOM += pJ.position * w;
                        sameTypeWeight += w;
                    }

                    // Inter-type repulsion
                    float2 repForce = float2.zero;
                    if (!sameType)
                    {
                        float repFactor = 1f - r / smoothingRadius;
                        repForce = dir * interTypeRepulsion * repFactor * repFactor;
                    }

                    totalForce += pressForce + nearForce + viscForce + cohesionForce + repForce;
                }
            }

            // Surface tension
            if (sameTypeWeight > 0.001f)
            {
                float2 com = sameTypeCOM / sameTypeWeight;
                float2 toCOM = com - pI.position;
                float distCOM = math.length(toCOM);
                if (distCOM > 0.001f)
                {
                    totalForce += (toCOM / distCOM) * surfaceTensionStrength * typeI.cohesion
                                * math.min(distCOM, smoothingRadius);
                }
            }

            forces[i] = totalForce;
        }

        int2 CellCoord(float2 pos)
        {
            return math.clamp(
                (int2)math.floor((pos - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1));
        }

        float SpikyGrad(float r)
        {
            if (r >= smoothingRadius || r < 1e-6f) return 0f;
            float diff = smoothingRadius - r;
            return spikyGradCoeff * diff * diff;
        }

        float ViscLaplacian(float r)
        {
            if (r >= smoothingRadius || r < 1e-6f) return 0f;
            return viscLaplCoeff * (smoothingRadius - r);
        }
    }

    // ─── Job 4: Integration + Boundaries + Suction ───────────────

    [BurstCompile]
    struct IntegrateJob : IJobParallelFor
    {
        public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<float2> forces;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<FluidTypeGPU> fluidTypeData;

        public float2 gravity;
        public float dt, velocityDamping, maxSpeed;
        public float particleRadius, boundaryDamping;
        public float2 containerMin, containerMax;
        public bool uniformFluid;

        public bool flaskActive;
        public float2 flaskPos;
        public int flaskTargetType;
        public float flaskRadius, flaskAbsorbRadius, flaskStrength;

        public void Execute(int i)
        {
            var p = particles[i];
            if (p.alive < 0.5f) return;

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
                        float pullFactor = 1f - dist / flaskRadius;
                        p.velocity += dir * flaskStrength * pullFactor * pullFactor * dt;

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
}