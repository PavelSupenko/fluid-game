using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;

/// <summary>
/// CPU-based incompressible SPH fluid simulation with Jobs + Burst.
///
/// INCOMPRESSIBILITY MODEL (4 layers):
///   1. Bidirectional pressure — negative pressure (tension) prevents density drops
///   2. PBF density correction — iterative position fix to enforce ρ = ρ₀
///   3. XSPH velocity smoothing — neighbors move together (thick paint effect)
///   4. Rest-neighbor springs — structural memory, breakable for deformation
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

    // ─── Feature 1: Incompressibility — Bidirectional Pressure ──
    [Header("Incompressibility")]
    [Tooltip("How much negative pressure (tension) is allowed. " +
             "0 = old behavior (compressible). 1 = full tension (incompressible).")]
    [Range(0f, 1f)]
    public float tensionFactor = 0.8f;

    [Tooltip("Maximum negative pressure as a fraction of pressureStiffness. " +
             "Prevents extreme tension from tearing particles apart.")]
    [Range(0f, 1f)]
    public float tensionClamp = 0.6f;

    // ─── Feature 2: Position-Based Density Correction (PBF) ─────
    [Header("Density Correction (PBF)")]
    [Tooltip("Enable iterative position correction to enforce constant density. " +
             "This is the main mechanism preventing column collapse.")]
    public bool enableDensityCorrection = true;

    [Tooltip("Number of correction iterations per substep. " +
             "More = more accurate but slower. 2-3 is good for mobile.")]
    [Range(1, 5)]
    public int densityCorrectionIterations = 2;

    [Tooltip("Relaxation parameter (epsilon). Prevents division by zero " +
             "and controls correction stiffness. Lower = stiffer.")]
    public float pbfRelaxation = 100f;

    // ─── Feature 3: XSPH Velocity Smoothing ─────────────────────
    [Header("XSPH Smoothing")]
    [Tooltip("Blend factor for velocity smoothing with neighbors. " +
             "0 = no smoothing. 0.3 = thick paint. 0.5 = very cohesive.")]
    [Range(0f, 0.5f)]
    public float xsphFactor = 0.2f;

    [Tooltip("Only smooth velocities with same-type neighbors. " +
             "Keeps different fluid types from dragging each other.")]
    public bool xsphSameTypeOnly = true;

    // ─── Feature 4: Rest-Neighbor Spring Network ─────────────────
    [Header("Spring Network")]
    [Tooltip("Enable spring network for structural memory. " +
             "Particles remember neighbors and resist deformation.")]
    public bool enableSprings = true;

    [Tooltip("Spring stiffness. Higher = more rigid structure.")]
    [Range(0f, 50f)]
    public float springStiffness = 15f;

    [Tooltip("Springs break when stretched beyond this factor of rest distance. " +
             "1.5 = 50% stretch before break. Lower = more brittle.")]
    [Range(1.1f, 3f)]
    public float springBreakThreshold = 1.8f;

    [Tooltip("Maximum springs per particle. Lower = faster, less structural memory.")]
    [Range(4, 20)]
    public int maxSpringsPerParticle = 12;

    [Tooltip("Spring damping to prevent oscillation.")]
    [Range(0f, 1f)]
    public float springDamping = 0.3f;

    // ─── Cohesion & Separation ───────────────────────────────────
    [Header("Cohesion & Separation")]
    [Range(0f, 50f)]
    public float cohesionStrength = 15f;
    [Range(0f, 30f)]
    public float interTypeRepulsion = 8f;
    [Range(0f, 30f)]
    public float surfaceTensionStrength = 5f;
    public bool uniformFluid = false;

    // ─── Particle Collision ────────────────────────────────────────
    [Header("Particle Collision")]
    [Range(0.3f, 5f)]
    public float collisionRadiusFactor = 0.85f;
    [Range(0.01f, 1.0f)]
    public float collisionPushStrength = 0.8f;

    // ─── Particle Merging (LOD) ──────────────────────────────────
    [Header("Particle Merging")]
    public bool enableMerging = true;
    [Range(2, 1_000)]
    public int maxMergeSize = 4;
    [Range(1, 30)]
    public int mergeInterval = 10;
    public float mergeVelocityThreshold = 0.1f;
    [Range(1f, 2.5f)]
    public float cohesionMassExponent = 1.5f;

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
    private NativeArray<float2> collisionCorrections;

    // Per-particle mass (separate from struct to keep 48-byte GPU layout)
    private NativeArray<float> particleMasses;

    // Sleep state
    private NativeArray<int> sleepState;
    private NativeArray<int> sleepCounter;

    // Spatial hash arrays
    private NativeArray<int> cellCounts;
    private NativeArray<int> cellOffsets;
    private NativeArray<int> sortedIndices;
    private NativeArray<int> particleCellIndex;

    // Feature 2: PBF density correction
    private NativeArray<float> lambdas;
    private NativeArray<float2> densityCorrections;

    // Feature 3: XSPH smoothed velocities
    private NativeArray<float2> smoothedVelocities;

    // Feature 4: Spring network
    // Packed as flat array: particle i's springs start at i * maxSpringsPerParticle
    // Each spring stores: neighbor index (int packed as float) + rest distance
    private NativeArray<int> springNeighbors;      // [i * maxSprings + k] = neighbor index (-1 = empty)
    private NativeArray<float> springRestLengths;   // [i * maxSprings + k] = rest distance
    private NativeArray<float2> springForces;       // Accumulated spring force per particle
    private bool springsBuilt = false;

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
        if (imageSource != null)
            imageSource.TryParseImage();

        if (imageSource != null && imageSource.IsReady)
            InitFromImage(imageSource);
        else
            SpawnParticles();

        InitNativeData();
    }

    private bool hasInteracted = false;
    private int previousAliveCount;

    void FixedUpdate()
    {
        float dt = (Time.fixedDeltaTime * timeScale) / subSteps;

        for (int step = 0; step < subSteps; step++)
            RunSimulationStep(dt);

        if (flaskActive)
        {
            WakeNearPoint(flaskPos, wakeRadius);
            WakeColumnAbove(flaskPos);
        }

        if (!hasInteracted)
        {
            int currentAlive = 0;
            for (int i = 0; i < ParticleCount; i++)
                if (particles[i].alive > 0.5f) currentAlive++;

            if (currentAlive < previousAliveCount)
            {
                hasInteracted = true;
                Debug.Log($"[FluidSimJobs] First absorption detected — wake systems activated");
            }
            previousAliveCount = currentAlive;
        }

        if (hasInteracted)
        {
            CascadeWake();
            DetectFloatingIslands();
            MergeParticles();
        }

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
        uniformFluid = false;
        startSleeping = true;

        float s = particleSpacing;
        smoothingRadius = s * 2.5f;
        particleRadius = s * 0.4f;
        nearPressureStiffness = 15f;
        pressureStiffness = 200f;
        subSteps = Mathf.Max(subSteps, 4);
        wakeRadius = smoothingRadius * 4f;

        cohesionStrength = 20f;
        interTypeRepulsion = 12f;
        surfaceTensionStrength = 8f;

        Debug.Log($"[FluidSimJobs] From image: {ParticleCount} particles, " +
                  $"spacing={s:F4}, h={smoothingRadius:F4}, incompressible mode");
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
        collisionCorrections = new NativeArray<float2>(ParticleCount, Allocator.Persistent);
        particleMasses = new NativeArray<float>(ParticleCount, Allocator.Persistent);
        for (int i = 0; i < ParticleCount; i++) particleMasses[i] = 1f;
        sleepState = new NativeArray<int>(ParticleCount, Allocator.Persistent);
        sleepCounter = new NativeArray<int>(ParticleCount, Allocator.Persistent);

        cellCounts = new NativeArray<int>(hashGridTotal, Allocator.Persistent);
        cellOffsets = new NativeArray<int>(hashGridTotal, Allocator.Persistent);
        sortedIndices = new NativeArray<int>(ParticleCount, Allocator.Persistent);
        particleCellIndex = new NativeArray<int>(ParticleCount, Allocator.Persistent);

        // Feature 2: PBF arrays
        lambdas = new NativeArray<float>(ParticleCount, Allocator.Persistent);
        densityCorrections = new NativeArray<float2>(ParticleCount, Allocator.Persistent);

        // Feature 3: XSPH array
        smoothedVelocities = new NativeArray<float2>(ParticleCount, Allocator.Persistent);

        // Feature 4: Spring network arrays
        int springArraySize = ParticleCount * maxSpringsPerParticle;
        springNeighbors = new NativeArray<int>(springArraySize, Allocator.Persistent);
        springRestLengths = new NativeArray<float>(springArraySize, Allocator.Persistent);
        springForces = new NativeArray<float2>(ParticleCount, Allocator.Persistent);
        // Initialize springs as empty
        for (int i = 0; i < springArraySize; i++)
        {
            springNeighbors[i] = -1;
            springRestLengths[i] = 0f;
        }

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

            sleepState[i] = (startSleeping && p.alive > 0.5f) ? 1 : 0;
            sleepCounter[i] = startSleeping ? sleepFramesRequired : 0;
        }

        if (autoRestDensity) CalibrateRestDensity();

        particleBuffer = new ComputeBuffer(ParticleCount, 48);
        UploadToGPU();

        AwakeCount = startSleeping ? 0 : ParticleCount;
        previousAliveCount = ParticleCount;
        Debug.Log($"[FluidSimJobs] Init: {ParticleCount} particles, " +
                  $"grid {hashGridW}x{hashGridH}, rest={restDensity:F1}, " +
                  $"awake={AwakeCount}, sleep={startSleeping}");
    }

    /// <summary>
    /// Feature 4: Builds the spring network from current particle positions.
    /// Called once after the first interaction wakes particles, or can be
    /// called manually. Each particle records its closest same-type neighbors
    /// and the current distance as rest length.
    /// </summary>
    void BuildSpringNetwork()
    {
        if (springsBuilt || !enableSprings) return;

        // Need a fresh spatial hash build first
        var assignJob = new AssignCellsJob
        {
            particles = particles,
            particleCellIndex = particleCellIndex,
            cellSize = cellSize,
            containerMin = new float2(containerMin.x, containerMin.y),
            gridW = hashGridW, gridH = hashGridH
        };
        assignJob.Schedule(ParticleCount, 128).Complete();

        new BuildGridJob
        {
            particleCellIndex = particleCellIndex,
            cellCounts = cellCounts,
            cellOffsets = cellOffsets,
            sortedIndices = sortedIndices,
            particleCount = ParticleCount,
            gridTotal = hashGridTotal
        }.Run();

        float hSqr = smoothingRadius * smoothingRadius;
        float2 cMin = new float2(containerMin.x, containerMin.y);

        for (int i = 0; i < ParticleCount; i++)
        {
            if (particles[i].alive < 0.5f) continue;

            float2 posI = particles[i].position;
            int typeI = particles[i].typeIndex;
            int2 cellI = math.clamp(
                (int2)math.floor((posI - cMin) / cellSize),
                int2.zero, new int2(hashGridW - 1, hashGridH - 1));

            int springBase = i * maxSpringsPerParticle;
            int springCount = 0;

            // Collect candidates sorted by distance
            // Use a simple insertion sort since maxSprings is small
            var candidateIdx = new NativeArray<int>(64, Allocator.Temp);
            var candidateDist = new NativeArray<float>(64, Allocator.Temp);
            int numCandidates = 0;

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                int2 nc = cellI + new int2(dx, dy);
                if (nc.x < 0 || nc.x >= hashGridW || nc.y < 0 || nc.y >= hashGridH) continue;

                int ci = nc.y * hashGridW + nc.x;
                int start = cellOffsets[ci];
                int count = cellCounts[ci];

                for (int s = 0; s < count; s++)
                {
                    int j = sortedIndices[start + s];
                    if (j == i) continue;
                    if (particles[j].alive < 0.5f) continue;
                    // Only connect same-type particles
                    if (particles[j].typeIndex != typeI) continue;

                    float dSqr = math.lengthsq(posI - particles[j].position);
                    if (dSqr >= hSqr) continue;

                    if (numCandidates < 64)
                    {
                        candidateIdx[numCandidates] = j;
                        candidateDist[numCandidates] = dSqr;
                        numCandidates++;
                    }
                }
            }

            // Pick closest maxSpringsPerParticle neighbors
            for (int k = 0; k < maxSpringsPerParticle && k < numCandidates; k++)
            {
                // Find minimum in remaining candidates
                int minIdx = k;
                for (int m = k + 1; m < numCandidates; m++)
                {
                    if (candidateDist[m] < candidateDist[minIdx])
                        minIdx = m;
                }
                // Swap
                if (minIdx != k)
                {
                    int tmpI = candidateIdx[k]; candidateIdx[k] = candidateIdx[minIdx]; candidateIdx[minIdx] = tmpI;
                    float tmpD = candidateDist[k]; candidateDist[k] = candidateDist[minIdx]; candidateDist[minIdx] = tmpD;
                }

                springNeighbors[springBase + k] = candidateIdx[k];
                springRestLengths[springBase + k] = math.sqrt(candidateDist[k]);
                springCount++;
            }

            // Mark remaining slots as empty
            for (int k = springCount; k < maxSpringsPerParticle; k++)
            {
                springNeighbors[springBase + k] = -1;
            }

            candidateIdx.Dispose();
            candidateDist.Dispose();
        }

        springsBuilt = true;
        Debug.Log($"[FluidSimJobs] Spring network built for {ParticleCount} particles, " +
                  $"max {maxSpringsPerParticle} springs each");
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

    private int cascadeScanCursor = 0;

    [Header("Scan Budget")]
    public int scanBudgetPerFrame = 400;

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

            if (sleepState[i] != 0) continue;
            if (particles[i].alive < 0.5f) continue;
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
                        sleepState[j] = 0;
                        sleepCounter[j] = 0;
                        wakeBudget--;
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
            int start = cellOffsets[ci];
            int count = cellCounts[ci];

            for (int s = 0; s < count && wakeBudget > 0; s++)
            {
                int i = sortedIndices[start + s];
                if (sleepState[i] == 0) continue;
                if (particles[i].alive < 0.5f) continue;

                bool typeMatch = (flaskTargetType < 0) || (particles[i].typeIndex == flaskTargetType);
                if (!typeMatch) continue;

                if (math.lengthsq(particles[i].position - point) < radiusSqr)
                {
                    sleepState[i] = 0;
                    sleepCounter[i] = 0;
                    wakeBudget--;
                }
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
            int start = cellOffsets[ci];
            int count = cellCounts[ci];

            for (int s = 0; s < count && wakeBudget > 0; s++)
            {
                int i = sortedIndices[start + s];
                if (sleepState[i] != 1) continue;
                if (particles[i].alive < 0.5f) continue;

                bool typeMatch = (flaskTargetType < 0) || (particles[i].typeIndex == flaskTargetType);
                if (!typeMatch) continue;

                float dxVal = particles[i].position.x - point.x;
                if (dxVal * dxVal < halfWidth * halfWidth)
                {
                    sleepState[i] = 0;
                    sleepCounter[i] = 0;
                    wakeBudget--;
                }
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
        {
            for (int x = 0; x < hashGridW; x++)
            {
                int ci = y * hashGridW + x;
                if (cellCounts[ci] == 0) { grounded[ci] = false; continue; }

                bool supported = false;
                int yBelow = y - 1;
                if (x > 0) supported |= grounded[yBelow * hashGridW + (x - 1)];
                supported |= grounded[yBelow * hashGridW + x];
                if (x < hashGridW - 1) supported |= grounded[yBelow * hashGridW + (x + 1)];
                grounded[ci] = supported;
            }
        }

        for (int ci = 0; ci < hashGridTotal && wakeBudget > 0; ci++)
        {
            if (grounded[ci]) continue;
            if (cellCounts[ci] == 0) continue;

            int start = cellOffsets[ci];
            int count = cellCounts[ci];
            for (int s = 0; s < count && wakeBudget > 0; s++)
            {
                int i = sortedIndices[start + s];
                if (sleepState[i] != 1) continue;
                sleepState[i] = 0;
                sleepCounter[i] = 0;
                wakeBudget--;
            }
        }

        // Slope flow
        for (int y = 1; y < hashGridH && wakeBudget > 0; y++)
        for (int x = 0; x < hashGridW && wakeBudget > 0; x++)
        {
            int ci = y * hashGridW + x;
            if (!grounded[ci]) continue;
            if (cellCounts[ci] == 0) continue;
            int belowCi = (y - 1) * hashGridW + x;
            if (cellCounts[belowCi] > 0) continue;
            WakeSleepingInCell(ci, ref wakeBudget);
        }

        // Horizontal spreading
        for (int y = 1; y < hashGridH && wakeBudget > 0; y++)
        for (int x = 0; x < hashGridW && wakeBudget > 0; x++)
        {
            int ci = y * hashGridW + x;
            if (!grounded[ci]) continue;
            if (cellCounts[ci] == 0) continue;

            bool flowLeft = false;
            if (x > 0)
            {
                bool sideEmpty = cellCounts[y * hashGridW + (x - 1)] == 0;
                bool diagEmpty = cellCounts[(y - 1) * hashGridW + (x - 1)] == 0;
                flowLeft = sideEmpty && diagEmpty;
            }
            bool flowRight = false;
            if (x < hashGridW - 1)
            {
                bool sideEmpty = cellCounts[y * hashGridW + (x + 1)] == 0;
                bool diagEmpty = cellCounts[(y - 1) * hashGridW + (x + 1)] == 0;
                flowRight = sideEmpty && diagEmpty;
            }
            if (flowLeft || flowRight)
                WakeSleepingInCell(ci, ref wakeBudget);
        }
    }

    void WakeSleepingInCell(int ci, ref int wakeBudget)
    {
        int start = cellOffsets[ci];
        int count = cellCounts[ci];
        for (int s = 0; s < count && wakeBudget > 0; s++)
        {
            int i = sortedIndices[start + s];
            if (sleepState[i] != 1) continue;
            sleepState[i] = 0;
            sleepCounter[i] = 0;
            wakeBudget--;
        }
    }

    int2 CellCoord(float2 pos)
    {
        return math.clamp(
            (int2)math.floor((pos - new float2(containerMin.x, containerMin.y)) / cellSize),
            int2.zero, new int2(hashGridW - 1, hashGridH - 1));
    }

    // ═════════════════════════════════════════════════════════════
    //  SIMULATION STEP — now includes PBF, XSPH, and Springs
    // ═════════════════════════════════════════════════════════════

    void RunSimulationStep(float dt)
    {
        // Build spring network on first wake (once only)
        if (enableSprings && !springsBuilt && hasInteracted)
            BuildSpringNetwork();

        // 1. Build spatial hash — ALL alive particles (sleeping + awake)
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

        // 2. Density — with bidirectional pressure (Feature 1)
        var densityJob = new DensityJob
        {
            particles = particles,
            sleepState = sleepState,
            sortedIndices = sortedIndices,
            cellCounts = cellCounts,
            cellOffsets = cellOffsets,
            particleMasses = particleMasses,
            densities = densities,
            pressures = pressures,
            cellSize = cellSize,
            containerMin = new float2(containerMin.x, containerMin.y),
            gridW = hashGridW, gridH = hashGridH,
            smoothingRadiusSqr = smoothingRadius * smoothingRadius,
            restDensity = restDensity,
            pressureStiffness = pressureStiffness,
            poly6Coeff = 4f / (math.PI * math.pow(smoothingRadius, 8)),
            // Feature 1 params
            tensionFactor = tensionFactor,
            tensionClamp = tensionClamp
        };
        densityJob.Schedule(ParticleCount, 128).Complete();

        // 3. Forces — pressure + viscosity + cohesion + inter-type repulsion
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
            particleMasses = particleMasses,
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
            cohesionMassExponent = cohesionMassExponent,
            uniformFluid = uniformFluid,
            spikyGradCoeff = -10f / (math.PI * math.pow(smoothingRadius, 5)),
            viscLaplCoeff = 40f / (math.PI * math.pow(smoothingRadius, 5))
        };
        forcesJob.Schedule(ParticleCount, 64).Complete();

        // 3b. Feature 4: Spring forces — structural memory
        if (enableSprings && springsBuilt)
        {
            new SpringForceJob
            {
                particles = particles,
                sleepState = sleepState,
                springNeighbors = springNeighbors,
                springRestLengths = springRestLengths,
                springForces = springForces,
                maxSpringsPerParticle = maxSpringsPerParticle,
                springStiffness = springStiffness,
                springDamping = springDamping,
                springBreakThreshold = springBreakThreshold
            }.Schedule(ParticleCount, 64).Complete();

            // Add spring forces to main forces
            new AddSpringForcesJob
            {
                forces = forces,
                springForces = springForces,
                sleepState = sleepState,
                alive = particles
            }.Schedule(ParticleCount, 256).Complete();
        }

        // 4. Integrate — velocity + position update
        var integrateJob = new IntegrateJob
        {
            particles = particles,
            sleepState = sleepState,
            forces = forces,
            densities = densities,
            particleMasses = particleMasses,
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

        // 5. Feature 2: Position-Based Density Correction (PBF)
        //    Iteratively nudge particles to restore rest density.
        //    This is the main incompressibility enforcement.
        if (enableDensityCorrection)
        {
            float poly6C = 4f / (math.PI * math.pow(smoothingRadius, 8));
            float spikyC = -10f / (math.PI * math.pow(smoothingRadius, 5));

            for (int iter = 0; iter < densityCorrectionIterations; iter++)
            {
                // Rebuild spatial hash with corrected positions
                if (iter > 0)
                {
                    assignJob.Schedule(ParticleCount, 128).Complete();
                    buildGridJob.Run();
                }

                // Compute lambda (constraint magnitude) for each particle
                new ComputeLambdaJob
                {
                    particles = particles,
                    sleepState = sleepState,
                    sortedIndices = sortedIndices,
                    cellCounts = cellCounts,
                    cellOffsets = cellOffsets,
                    particleMasses = particleMasses,
                    lambdas = lambdas,
                    cellSize = cellSize,
                    containerMin = new float2(containerMin.x, containerMin.y),
                    gridW = hashGridW, gridH = hashGridH,
                    smoothingRadius = smoothingRadius,
                    smoothingRadiusSqr = smoothingRadius * smoothingRadius,
                    restDensity = restDensity,
                    poly6Coeff = poly6C,
                    spikyGradCoeff = spikyC,
                    relaxation = pbfRelaxation,
                    particleMass = particleMass
                }.Schedule(ParticleCount, 128).Complete();

                // Compute position corrections (read-only particles)
                new ComputeDensityCorrectionJob
                {
                    particles = particles,
                    sleepState = sleepState,
                    sortedIndices = sortedIndices,
                    cellCounts = cellCounts,
                    cellOffsets = cellOffsets,
                    particleMasses = particleMasses,
                    lambdas = lambdas,
                    densityCorrections = densityCorrections,
                    cellSize = cellSize,
                    containerMin = new float2(containerMin.x, containerMin.y),
                    gridW = hashGridW, gridH = hashGridH,
                    smoothingRadius = smoothingRadius,
                    smoothingRadiusSqr = smoothingRadius * smoothingRadius,
                    restDensity = restDensity,
                    spikyGradCoeff = spikyC,
                    particleMass = particleMass
                }.Schedule(ParticleCount, 128).Complete();

                // Apply corrections to positions (writes only particles[i])
                new ApplyDensityCorrectionJob
                {
                    particles = particles,
                    sleepState = sleepState,
                    densityCorrections = densityCorrections,
                    containerMin = new float2(containerMin.x, containerMin.y),
                    containerMax = new float2(containerMax.x, containerMax.y),
                    particleRadius = particleRadius
                }.Schedule(ParticleCount, 128).Complete();
            }
        }

        // 5b. Feature 3: XSPH velocity smoothing
        //     Blends each particle's velocity with neighbors for cohesive movement.
        if (xsphFactor > 0.001f)
        {
            new XSPHJob
            {
                particles = particles,
                sleepState = sleepState,
                sortedIndices = sortedIndices,
                cellCounts = cellCounts,
                cellOffsets = cellOffsets,
                densities = densities,
                particleMasses = particleMasses,
                smoothedVelocities = smoothedVelocities,
                cellSize = cellSize,
                containerMin = new float2(containerMin.x, containerMin.y),
                gridW = hashGridW, gridH = hashGridH,
                smoothingRadiusSqr = smoothingRadius * smoothingRadius,
                poly6Coeff = 4f / (math.PI * math.pow(smoothingRadius, 8)),
                xsphFactor = xsphFactor,
                sameTypeOnly = xsphSameTypeOnly
            }.Schedule(ParticleCount, 128).Complete();

            // Apply smoothed velocities
            new ApplyXSPHJob
            {
                particles = particles,
                sleepState = sleepState,
                smoothedVelocities = smoothedVelocities
            }.Schedule(ParticleCount, 256).Complete();
        }

        // 6. Particle collision — hard distance constraint
        new ParticleCollisionJob
        {
            particles = particles,
            sleepState = sleepState,
            sortedIndices = sortedIndices,
            cellCounts = cellCounts,
            cellOffsets = cellOffsets,
            particleMasses = particleMasses,
            collisionCorrections = collisionCorrections,
            cellSize = cellSize,
            containerMin = new float2(containerMin.x, containerMin.y),
            gridW = hashGridW, gridH = hashGridH,
            minDistance = particleSpacing * collisionRadiusFactor,
            pushStrength = collisionPushStrength
        }.Schedule(ParticleCount, 128).Complete();

        new ApplyCollisionJob
        {
            particles = particles,
            collisionCorrections = collisionCorrections,
            sleepState = sleepState
        }.Schedule(ParticleCount, 256).Complete();

        // 7. Sleep check
        new SleepCheckJob
        {
            particles = particles,
            sleepState = sleepState,
            sleepCounter = sleepCounter,
            sleepVelThresholdSqr = sleepVelocityThreshold * sleepVelocityThreshold,
            sleepFramesRequired = sleepFramesRequired
        }.Schedule(ParticleCount, 256).Complete();
    }

    // ─── GPU Upload ──────────────────────────────────────────────

    void UploadToGPU()
    {
        int awake = 0;
        for (int i = 0; i < ParticleCount; i++)
        {
            var p = particles[i];
            float renderMass = particleMasses[i];
            Particles[i] = new FluidParticle
            {
                position = new Vector2(p.position.x, p.position.y),
                velocity = new Vector2(p.velocity.x, p.velocity.y),
                typeIndex = p.typeIndex,
                density = renderMass,
                pressure = pressures[i],
                alive = p.alive,
                color = new Color(p.color.x, p.color.y, p.color.z, p.color.w)
            };
            if (p.alive > 0.5f && sleepState[i] == 0) awake++;
        }
        AwakeCount = awake;
        particleBuffer.SetData(Particles);
    }

    // ─── Particle Merging ────────────────────────────────────────

    void MergeParticles()
    {
        if (!enableMerging) return;
        if (Time.frameCount % mergeInterval != 0) return;

        float baseMergeDist = particleSpacing * 1.2f;
        float mergeVelSqr = mergeVelocityThreshold * mergeVelocityThreshold;
        int merged = 0;

        for (int i = 0; i < ParticleCount; i++)
        {
            if (particles[i].alive < 0.5f) continue;
            if (sleepState[i] == 1) continue;
            if (particleMasses[i] >= maxMergeSize) continue;
            if (math.lengthsq(particles[i].velocity) > mergeVelSqr) continue;

            float2 posI = particles[i].position;
            int typeI = particles[i].typeIndex;
            float radiusI = math.sqrt(particleMasses[i]);
            int2 cell = CellCoord(posI);

            int bestJ = -1;
            float bestDistSqr = float.MaxValue;

            for (int dx2 = -1; dx2 <= 1; dx2++)
            for (int dy2 = -1; dy2 <= 1; dy2++)
            {
                int2 nc = cell + new int2(dx2, dy2);
                if (nc.x < 0 || nc.x >= hashGridW || nc.y < 0 || nc.y >= hashGridH) continue;

                int ci = nc.y * hashGridW + nc.x;
                int start = cellOffsets[ci];
                int count = cellCounts[ci];

                for (int s = 0; s < count; s++)
                {
                    int j = sortedIndices[start + s];
                    if (j <= i) continue;
                    if (particles[j].alive < 0.5f) continue;
                    if (sleepState[j] == 1) continue;
                    if (particles[j].typeIndex != typeI) continue;
                    if (math.lengthsq(particles[j].velocity) > mergeVelSqr) continue;
                    if (particleMasses[i] + particleMasses[j] > maxMergeSize) continue;

                    float radiusJ = math.sqrt(particleMasses[j]);
                    float pairMergeDist = baseMergeDist * (radiusI + radiusJ) * 0.5f;
                    float dSqr = math.lengthsq(posI - particles[j].position);
                    if (dSqr < pairMergeDist * pairMergeDist && dSqr < bestDistSqr)
                    {
                        bestDistSqr = dSqr;
                        bestJ = j;
                    }
                }
            }

            if (bestJ < 0) continue;

            float massA = particleMasses[i];
            float massB = particleMasses[bestJ];
            float totalMass = massA + massB;

            var pI = particles[i];
            var pB = particles[bestJ];
            float2 newPos = (pI.position * massA + pB.position * massB) / totalMass;
            float2 newVel = (pI.velocity * massA + pB.velocity * massB) / totalMass;

            pI.position = newPos;
            pI.velocity = newVel;
            particles[i] = pI;
            particleMasses[i] = totalMass;

            pB.alive = 0f;
            pB.position = new float2(-9999, -9999);
            pB.velocity = float2.zero;
            particles[bestJ] = pB;

            merged++;
        }

        if (merged > 0)
            Debug.Log($"[FluidSimJobs] Merged {merged} particle pairs");
    }

    void DisposeNative()
    {
        if (particles.IsCreated) particles.Dispose();
        if (forces.IsCreated) forces.Dispose();
        if (densities.IsCreated) densities.Dispose();
        if (pressures.IsCreated) pressures.Dispose();
        if (collisionCorrections.IsCreated) collisionCorrections.Dispose();
        if (particleMasses.IsCreated) particleMasses.Dispose();
        if (sleepState.IsCreated) sleepState.Dispose();
        if (sleepCounter.IsCreated) sleepCounter.Dispose();
        if (cellCounts.IsCreated) cellCounts.Dispose();
        if (cellOffsets.IsCreated) cellOffsets.Dispose();
        if (sortedIndices.IsCreated) sortedIndices.Dispose();
        if (particleCellIndex.IsCreated) particleCellIndex.Dispose();
        if (fluidTypeData.IsCreated) fluidTypeData.Dispose();
        // Feature 2
        if (lambdas.IsCreated) lambdas.Dispose();
        if (densityCorrections.IsCreated) densityCorrections.Dispose();
        // Feature 3
        if (smoothedVelocities.IsCreated) smoothedVelocities.Dispose();
        // Feature 4
        if (springNeighbors.IsCreated) springNeighbors.Dispose();
        if (springRestLengths.IsCreated) springRestLengths.Dispose();
        if (springForces.IsCreated) springForces.Dispose();
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

    // ─── Density (parallel) — Feature 1: bidirectional pressure ──

    [BurstCompile]
    struct DensityJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<int> sortedIndices;
        [ReadOnly] public NativeArray<int> cellCounts;
        [ReadOnly] public NativeArray<int> cellOffsets;
        [ReadOnly] public NativeArray<float> particleMasses;
        [NativeDisableParallelForRestriction] public NativeArray<float> densities;
        [NativeDisableParallelForRestriction] public NativeArray<float> pressures;

        public float cellSize, smoothingRadiusSqr;
        public float2 containerMin;
        public int gridW, gridH;
        public float restDensity, pressureStiffness, poly6Coeff;

        // Feature 1: tension parameters
        public float tensionFactor;
        public float tensionClamp;

        public void Execute(int i)
        {
            if (sleepState[i] == 1 || particles[i].alive < 0.5f)
            {
                densities[i] = restDensity;
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

                for (int s = 0; s < count; s++)
                {
                    int j = sortedIndices[start + s];
                    float rSqr = math.lengthsq(posI - particles[j].position);
                    if (rSqr < smoothingRadiusSqr)
                    {
                        float diff = smoothingRadiusSqr - rSqr;
                        density += particleMasses[j] * poly6Coeff * diff * diff * diff;
                    }
                }
            }

            density = math.max(density, 0.001f);
            densities[i] = density;

            // ── Feature 1: Bidirectional pressure ──
            // OLD: pressures[i] = math.max(0f, pressureStiffness * (density - restDensity));
            //
            // NEW: Allow negative pressure (tension) when density < restDensity.
            // This creates a pulling force that prevents particles from separating,
            // which is the key mechanism for volume preservation.
            //
            // tensionFactor controls how much negative pressure is applied:
            //   0.0 = old behavior (compressible, no tension)
            //   1.0 = full symmetric pressure (maximally incompressible)
            //
            // tensionClamp limits the maximum negative pressure to prevent
            // extreme tension from tearing the material apart.

            float pressureError = density - restDensity;
            float pressure;

            if (pressureError >= 0f)
            {
                // Compression: full repulsive pressure (unchanged from original)
                pressure = pressureStiffness * pressureError;
            }
            else
            {
                // Tension: negative pressure pulls particles together
                // Scaled by tensionFactor and clamped to prevent instability
                pressure = pressureStiffness * pressureError * tensionFactor;
                float maxTension = -pressureStiffness * restDensity * tensionClamp;
                pressure = math.max(pressure, maxTension);
            }

            pressures[i] = pressure;
        }
    }

    // ─── Forces (parallel) ───────────────────────────────────────

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
        [ReadOnly] public NativeArray<float> particleMasses;

        public float cellSize, smoothingRadius, smoothingRadiusSqr, particleMass;
        public float2 containerMin;
        public int gridW, gridH;
        public float nearPressureStiffness;
        public float cohesionStrength;
        public float interTypeRepulsion;
        public float surfaceTensionStrength;
        public float cohesionMassExponent;
        public bool uniformFluid;
        public float spikyGradCoeff, viscLaplCoeff;

        public void Execute(int i)
        {
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
                    float densityJ = math.max(densities[j], 0.001f);

                    // Pressure force — now supports negative pressure (tension)
                    // from Feature 1. When pressures are negative, force direction
                    // reverses: particles pull toward each other.
                    float gm = SpikyGrad(r);
                    float pressAvg = (pressures[i] + pressures[j]) * 0.5f;
                    float2 pF = dir * (-particleMass * pressAvg * gm / densityJ);

                    // Near-pressure repulsion (unchanged)
                    float nf = 1f - r / smoothingRadius;
                    float2 nF = dir * (nearPressureStiffness * nf * nf);

                    // Viscosity (unchanged)
                    var typeJ = fluidTypeData[pJ.typeIndex];
                    float mu = (typeI.viscosity + typeJ.viscosity) * 0.5f;
                    float vl = ViscLapl(r);
                    float2 vF = mu * particleMass * (pJ.velocity - pI.velocity)
                              / densityJ * vl;

                    bool sameType = (pI.typeIndex == pJ.typeIndex);

                    // Cohesion — same type only
                    float2 cF = float2.zero;
                    if (sameType)
                    {
                        float t = r / smoothingRadius;
                        float massScale = math.min(
                            math.pow(particleMasses[j], cohesionMassExponent),
                            20f
                        );
                        cF = -dir * typeI.cohesion * cohesionStrength * massScale * t * (1f - t) * (1f - t);

                        float w = (1f - t) * math.min(particleMasses[j], 10f);
                        sameTypeCOM += pJ.position * w;
                        sameTypeWeight += w;
                    }

                    // Inter-type repulsion
                    float2 rF = float2.zero;
                    if (!sameType && interTypeRepulsion > 0f)
                    {
                        float rf = 1f - r / smoothingRadius;
                        rF = dir * interTypeRepulsion * rf * rf;
                    }

                    totalForce += pF + nF + vF + cF + rF;
                }
            }

            // Surface tension
            if (sameTypeWeight > 0.001f && surfaceTensionStrength > 0f)
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

    // ─── Integrate (parallel) ────────────────────────────────────

    [BurstCompile]
    struct IntegrateJob : IJobParallelFor
    {
        public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<float2> forces;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<float> particleMasses;
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
            if (p.alive < 0.5f || sleepState[i] == 1) return;

            var ft = fluidTypeData[p.typeIndex];
            float gScale = uniformFluid ? 1f : ft.gravityScale;
            float density = math.max(densities[i], 0.001f);
            float mass = particleMasses[i];

            float2 accel = forces[i] / (density * math.sqrt(mass)) + gravity * gScale;

            p.velocity += accel * dt;

            float massDamping = 1f / (1f + (mass - 1f) * 0.005f);
            p.velocity *= massDamping;

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

            // Progressive settling
            float settleThreshold = 0.5f;
            if (speedSqr < settleThreshold * settleThreshold && speedSqr > 1e-8f)
            {
                float speed = math.sqrt(speedSqr);
                float settleFactor = speed / settleThreshold;
                float extraDamp = math.lerp(0.85f, 1.0f, settleFactor);
                p.velocity *= extraDamp;
                speedSqr = math.lengthsq(p.velocity);
            }

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

    // ═════════════════════════════════════════════════════════════
    //  Feature 2: PBF DENSITY CORRECTION JOBS
    // ═════════════════════════════════════════════════════════════

    /// <summary>
    /// Computes the PBF lambda (Lagrange multiplier) for each particle.
    /// Lambda measures how much position correction is needed to satisfy
    /// the density constraint ρ_i = ρ₀.
    ///
    /// From Macklin & Müller 2013 "Position Based Fluids":
    ///   λ_i = -(C_i) / (Σ_k |∇_k C_i|² + ε)
    /// where C_i = ρ_i/ρ₀ - 1 is the density constraint.
    /// </summary>
    [BurstCompile]
    struct ComputeLambdaJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<int> sortedIndices;
        [ReadOnly] public NativeArray<int> cellCounts;
        [ReadOnly] public NativeArray<int> cellOffsets;
        [ReadOnly] public NativeArray<float> particleMasses;
        [NativeDisableParallelForRestriction]
        public NativeArray<float> lambdas;

        public float cellSize, smoothingRadius, smoothingRadiusSqr;
        public float2 containerMin;
        public int gridW, gridH;
        public float restDensity, poly6Coeff, spikyGradCoeff;
        public float relaxation, particleMass;

        public void Execute(int i)
        {
            if (sleepState[i] == 1 || particles[i].alive < 0.5f)
            {
                lambdas[i] = 0f;
                return;
            }

            float2 posI = particles[i].position;
            int2 cellI = math.clamp(
                (int2)math.floor((posI - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1));

            // Recompute density at current position (may have been corrected)
            float density = 0f;
            float2 gradSum = float2.zero;    // Σ ∇_pk W for k=j (neighbor contributions)
            float gradSqrSum = 0f;           // Σ |∇_pk C_i|² for all k

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
                    float2 diff = posI - particles[j].position;
                    float rSqr = math.lengthsq(diff);
                    if (rSqr >= smoothingRadiusSqr) continue;

                    // Density contribution (Poly6)
                    float d = smoothingRadiusSqr - rSqr;
                    density += particleMasses[j] * poly6Coeff * d * d * d;

                    if (j == i) continue;
                    if (rSqr < 1e-12f) continue;

                    // Gradient of constraint w.r.t. neighbor j (Spiky gradient)
                    float r = math.sqrt(rSqr);
                    if (r >= smoothingRadius) continue;

                    float hMinusR = smoothingRadius - r;
                    float spikyGrad = spikyGradCoeff * hMinusR * hMinusR;
                    float2 gradJ = (diff / r) * (particleMass / restDensity) * spikyGrad;

                    // Accumulate for k=i (self) gradient
                    gradSum += gradJ;

                    // |∇_pj C_i|² for k=j
                    gradSqrSum += math.lengthsq(gradJ);
                }
            }

            // |∇_pi C_i|² (self gradient is the sum of all neighbor gradients)
            gradSqrSum += math.lengthsq(gradSum);

            // Density constraint: C_i = ρ_i/ρ₀ - 1
            float constraint = density / restDensity - 1f;

            // Lambda: correction magnitude
            lambdas[i] = -constraint / (gradSqrSum + relaxation);
        }
    }

    /// <summary>
    /// Computes position corrections based on lambda values (read-only particles).
    /// Double-buffered: writes corrections to a separate array, applied later.
    ///   Δp_i = (1/ρ₀) Σ_j (λ_i + λ_j) ∇W(p_i - p_j)
    /// </summary>
    [BurstCompile]
    struct ComputeDensityCorrectionJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<int> sortedIndices;
        [ReadOnly] public NativeArray<int> cellCounts;
        [ReadOnly] public NativeArray<int> cellOffsets;
        [ReadOnly] public NativeArray<float> particleMasses;
        [ReadOnly] public NativeArray<float> lambdas;
        [NativeDisableParallelForRestriction]
        public NativeArray<float2> densityCorrections;

        public float cellSize, smoothingRadius, smoothingRadiusSqr;
        public float2 containerMin;
        public int gridW, gridH;
        public float restDensity, spikyGradCoeff, particleMass;

        public void Execute(int i)
        {
            if (sleepState[i] == 1 || particles[i].alive < 0.5f)
            {
                densityCorrections[i] = float2.zero;
                return;
            }

            float2 posI = particles[i].position;
            int2 cellI = math.clamp(
                (int2)math.floor((posI - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1));

            float lambdaI = lambdas[i];
            float2 correction = float2.zero;

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
                    if (particles[j].alive < 0.5f) continue;

                    float2 diff = posI - particles[j].position;
                    float rSqr = math.lengthsq(diff);
                    if (rSqr >= smoothingRadiusSqr || rSqr < 1e-12f) continue;

                    float r = math.sqrt(rSqr);
                    if (r >= smoothingRadius) continue;

                    float hMinusR = smoothingRadius - r;
                    float spikyGrad = spikyGradCoeff * hMinusR * hMinusR;
                    float2 gradW = (diff / r) * spikyGrad;

                    // Position correction from combined lambdas
                    correction += (lambdaI + lambdas[j]) * gradW;
                }
            }

            densityCorrections[i] = correction * (particleMass / restDensity);
        }
    }

    /// <summary>
    /// Applies precomputed density corrections to particle positions.
    /// Separated from computation to avoid read/write race on particles array.
    /// </summary>
    [BurstCompile]
    struct ApplyDensityCorrectionJob : IJobParallelFor
    {
        public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<float2> densityCorrections;
        public float2 containerMin, containerMax;
        public float particleRadius;

        public void Execute(int i)
        {
            if (sleepState[i] == 1 || particles[i].alive < 0.5f) return;

            float2 corr = densityCorrections[i];
            if (math.lengthsq(corr) < 1e-14f) return;

            var p = particles[i];
            p.position += corr;

            // Re-enforce boundaries after correction
            float pr = particleRadius;
            p.position = math.clamp(p.position,
                containerMin + new float2(pr, pr),
                containerMax - new float2(pr, pr));

            particles[i] = p;
        }
    }

    // ═════════════════════════════════════════════════════════════
    //  Feature 3: XSPH VELOCITY SMOOTHING JOBS
    // ═════════════════════════════════════════════════════════════

    /// <summary>
    /// XSPH velocity smoothing: blends each particle's velocity with
    /// its neighbors weighted by the Poly6 kernel. This creates thick,
    /// paint-like movement where nearby particles move as a group.
    ///
    /// v_i_smoothed = v_i + ε Σ_j (m_j/ρ_j)(v_j - v_i) W(r_ij)
    ///
    /// ε (xsphFactor) controls smoothing strength:
    ///   0.0 = no effect
    ///   0.2 = thick paint
    ///   0.5 = very cohesive movement
    /// </summary>
    [BurstCompile]
    struct XSPHJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<int> sortedIndices;
        [ReadOnly] public NativeArray<int> cellCounts;
        [ReadOnly] public NativeArray<int> cellOffsets;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<float> particleMasses;
        [NativeDisableParallelForRestriction]
        public NativeArray<float2> smoothedVelocities;

        public float cellSize, smoothingRadiusSqr, poly6Coeff;
        public float2 containerMin;
        public int gridW, gridH;
        public float xsphFactor;
        public bool sameTypeOnly;

        public void Execute(int i)
        {
            if (sleepState[i] == 1 || particles[i].alive < 0.5f)
            {
                smoothedVelocities[i] = particles[i].velocity;
                return;
            }

            float2 posI = particles[i].position;
            float2 velI = particles[i].velocity;
            int typeI = particles[i].typeIndex;

            int2 cellI = math.clamp(
                (int2)math.floor((posI - containerMin) / cellSize),
                int2.zero, new int2(gridW - 1, gridH - 1));

            float2 velCorrection = float2.zero;

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
                    if (particles[j].alive < 0.5f) continue;

                    // Optional: only smooth with same type
                    if (sameTypeOnly && particles[j].typeIndex != typeI) continue;

                    float rSqr = math.lengthsq(posI - particles[j].position);
                    if (rSqr >= smoothingRadiusSqr) continue;

                    // Poly6 kernel weight
                    float diff = smoothingRadiusSqr - rSqr;
                    float w = poly6Coeff * diff * diff * diff;

                    float densJ = math.max(densities[j], 0.001f);
                    velCorrection += (particleMasses[j] / densJ)
                                   * (particles[j].velocity - velI) * w;
                }
            }

            smoothedVelocities[i] = velI + xsphFactor * velCorrection;
        }
    }

    /// <summary>
    /// Applies the smoothed velocities computed by XSPHJob.
    /// Separated into its own job so XSPHJob can read all velocities
    /// without race conditions.
    /// </summary>
    [BurstCompile]
    struct ApplyXSPHJob : IJobParallelFor
    {
        public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<float2> smoothedVelocities;

        public void Execute(int i)
        {
            if (sleepState[i] == 1 || particles[i].alive < 0.5f) return;

            var p = particles[i];
            p.velocity = smoothedVelocities[i];
            particles[i] = p;
        }
    }

    // ═════════════════════════════════════════════════════════════
    //  Feature 4: SPRING NETWORK JOBS
    // ═════════════════════════════════════════════════════════════

    /// <summary>
    /// Computes spring forces from the rest-neighbor network.
    /// Each particle has up to maxSpringsPerParticle connections to
    /// same-type neighbors, with remembered rest distances.
    ///
    /// Springs apply a Hookean force: F = k * (distance - restDistance) * dir
    /// Springs break when stretched beyond springBreakThreshold × restDistance,
    /// allowing the material to tear and deform permanently.
    ///
    /// Damping is applied along the spring axis to prevent oscillation:
    ///   F_damp = -damping * dot(relativeVelocity, dir) * dir
    /// </summary>
    [BurstCompile]
    struct SpringForceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [NativeDisableParallelForRestriction]
        public NativeArray<int> springNeighbors;
        [ReadOnly] public NativeArray<float> springRestLengths;
        [NativeDisableParallelForRestriction]
        public NativeArray<float2> springForces;

        public int maxSpringsPerParticle;
        public float springStiffness;
        public float springDamping;
        public float springBreakThreshold;

        public void Execute(int i)
        {
            if (sleepState[i] == 1 || particles[i].alive < 0.5f)
            {
                springForces[i] = float2.zero;
                return;
            }

            float2 posI = particles[i].position;
            float2 velI = particles[i].velocity;
            float2 totalForce = float2.zero;
            int baseIdx = i * maxSpringsPerParticle;

            for (int k = 0; k < maxSpringsPerParticle; k++)
            {
                int j = springNeighbors[baseIdx + k];
                if (j < 0) continue; // Empty slot
                if (particles[j].alive < 0.5f)
                {
                    // Neighbor was absorbed — break spring permanently
                    springNeighbors[baseIdx + k] = -1;
                    continue;
                }

                float restLen = springRestLengths[baseIdx + k];
                float2 diff = particles[j].position - posI;
                float dist = math.length(diff);

                if (dist < 1e-8f) continue;

                // Check if spring should break
                if (dist > restLen * springBreakThreshold)
                {
                    // Spring broken — permanent deformation
                    springNeighbors[baseIdx + k] = -1;
                    continue;
                }

                float2 dir = diff / dist;
                float stretch = dist - restLen;

                // Hookean spring force
                float2 springF = dir * springStiffness * stretch;

                // Axial damping to prevent oscillation
                float2 relVel = particles[j].velocity - velI;
                float axialVel = math.dot(relVel, dir);
                float2 dampF = dir * springDamping * axialVel;

                totalForce += springF + dampF;
            }

            springForces[i] = totalForce;
        }
    }

    /// <summary>
    /// Adds spring forces to the main force accumulator.
    /// Separated so SpringForceJob can safely read/write spring arrays.
    /// </summary>
    [BurstCompile]
    struct AddSpringForcesJob : IJobParallelFor
    {
        public NativeArray<float2> forces;
        [ReadOnly] public NativeArray<float2> springForces;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<ParticleData> alive;

        public void Execute(int i)
        {
            if (sleepState[i] == 1 || alive[i].alive < 0.5f) return;
            forces[i] = forces[i] + springForces[i];
        }
    }

    // ─── Particle collision (parallel) ───────────────────────────

    [BurstCompile]
    struct ParticleCollisionJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<int> sleepState;
        [ReadOnly] public NativeArray<int> sortedIndices;
        [ReadOnly] public NativeArray<int> cellCounts;
        [ReadOnly] public NativeArray<int> cellOffsets;
        [ReadOnly] public NativeArray<float> particleMasses;
        [NativeDisableParallelForRestriction]
        public NativeArray<float2> collisionCorrections;
        public float cellSize;
        public float2 containerMin;
        public int gridW, gridH;
        public float minDistance;
        public float pushStrength;

        public void Execute(int i)
        {
            if (particles[i].alive < 0.5f || sleepState[i] == 1)
            {
                collisionCorrections[i] = float2.zero;
                return;
            }

            float2 posI = particles[i].position;
            float radiusI = math.pow(particleMasses[i], 0.35f);

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
                int start = cellOffsets[ci];
                int count = cellCounts[ci];

                for (int s = 0; s < count; s++)
                {
                    int j = sortedIndices[start + s];
                    if (j == i) continue;
                    if (particles[j].alive < 0.5f) continue;

                    float radiusJ = math.pow(particleMasses[j], 0.35f);
                    float pairMinDist = minDistance * (radiusI + radiusJ) * 0.5f;

                    float2 diff = posI - particles[j].position;
                    float distSqr = math.lengthsq(diff);

                    if (distSqr < pairMinDist * pairMinDist && distSqr > 1e-12f)
                    {
                        float dist = math.sqrt(distSqr);
                        float overlap = pairMinDist - dist;
                        float2 dir = diff / dist;
                        totalPush += dir * overlap * pushStrength;
                        pushCount++;
                    }
                }
            }

            collisionCorrections[i] = (pushCount > 0) ? totalPush / pushCount : float2.zero;
        }
    }

    [BurstCompile]
    struct ApplyCollisionJob : IJobParallelFor
    {
        public NativeArray<ParticleData> particles;
        [ReadOnly] public NativeArray<float2> collisionCorrections;
        [ReadOnly] public NativeArray<int> sleepState;

        public void Execute(int i)
        {
            if (particles[i].alive < 0.5f || sleepState[i] == 1) return;

            float2 corr = collisionCorrections[i];
            if (math.lengthsq(corr) < 1e-12f) return;

            var p = particles[i];
            p.position += corr;
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
            if (sleepState[i] == 1 || particles[i].alive < 0.5f) return;

            float speedSqr = math.lengthsq(particles[i].velocity);

            if (speedSqr < sleepVelThresholdSqr)
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