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

    // ─── Particle Collision ────────────────────────────────────────
    [Header("Particle Collision")]
    [Tooltip("Minimum distance between particles as a factor of particleSpacing. " +
             "0.85 = particles can't get closer than 85% of their initial spacing.")]
    [Range(0.3f, 5f)]
    public float collisionRadiusFactor = 0.85f;

    [Tooltip("How strongly overlapping particles push apart per sub-step. " +
             "1.0 = full correction in one step (rigid). 0.5 = half correction (softer).")]
    [Range(0.01f, 1.0f)]
    public float collisionPushStrength = 0.8f;

    // ─── Particle Merging (LOD) ──────────────────────────────────
    [Header("Particle Merging")]
    [Tooltip("Enable merging of nearby same-type particles into larger droplets")]
    public bool enableMerging = true;

    [Tooltip("Max original particles that can merge into one large droplet")]
    [Range(2, 1_000)]
    public int maxMergeSize = 4;

    [Tooltip("How often to run the merge pass (every N frames). Lower = more responsive.")]
    [Range(1, 30)]
    public int mergeInterval = 10;

    [Tooltip("Particles merge only when their speed is below this threshold")]
    public float mergeVelocityThreshold = 0.1f;

    [Tooltip("Cohesion power exponent for merged particles. " +
             "1.0 = linear, 1.5 = super-linear (big drops attract more aggressively)")]
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
    private NativeArray<float2> collisionCorrections;

    // Per-particle mass (separate from struct to keep 48-byte GPU layout)
    // Default 1.0, increases when particles merge
    private NativeArray<float> particleMasses;

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
        if (imageSource != null) 
            imageSource.TryParseImage();
        
        if (imageSource != null && imageSource.IsReady)
            InitFromImage(imageSource);
        else
            SpawnParticles();

        InitNativeData();
    }

    // Track whether any particle has been absorbed (not just clicked)
    private bool hasInteracted = false;
    private int previousAliveCount;

    void FixedUpdate()
    {
        float dt = (Time.fixedDeltaTime * timeScale) / subSteps;

        for (int step = 0; step < subSteps; step++)
            RunSimulationStep(dt);

        // Wake near flask — always allowed (just wakes target-type particles,
        // they need to move toward flask to get absorbed)
        if (flaskActive)
        {
            WakeNearPoint(flaskPos, wakeRadius);
            WakeColumnAbove(flaskPos);
        }

        // Check if any particle was actually absorbed this frame
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

        // Aggressive wake systems — only after first real absorption
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
        uniformFluid = false; // Type-aware: cohesion by color, inter-type repulsion
        startSleeping = true;

        float s = particleSpacing;
        smoothingRadius = s * 2.5f;
        particleRadius = s * 0.4f;
        nearPressureStiffness = 15f;
        pressureStiffness = 200f;
        subSteps = Mathf.Max(subSteps, 4);
        wakeRadius = smoothingRadius * 4f;

        // Mercury-like behavior: same-type particles clump, different types separate
        cohesionStrength = 20f;          // Strong same-type attraction (mercury clumping)
        interTypeRepulsion = 12f;        // Different colors push apart (immiscible)
        surfaceTensionStrength = 8f;     // Droplets stay round, not flat

        Debug.Log($"[FluidSimJobs] From image: {ParticleCount} particles, " +
                  $"spacing={s:F4}, h={smoothingRadius:F4}, mercury mode");
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
        previousAliveCount = ParticleCount;
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

    // Rotating cursors — each method resumes scanning where it left off last frame.
    // This spreads the O(n) scan cost across multiple frames.
    private int cascadeScanCursor = 0;

    [Header("Scan Budget")]
    [Tooltip("Max particles to SCAN (not wake) per frame in cascade/unsupported checks. " +
             "Lower = less CPU per frame, but slower reaction to changes.")]
    public int scanBudgetPerFrame = 400;

    /// <summary>
    /// Moving particles wake sleeping neighbors.
    /// Scans a chunk of particles each frame using a rotating cursor.
    /// </summary>
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

            // Only awake, alive, moving particles can cascade-wake
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

    /// <summary>
    /// Wakes sleeping particles near the flask cursor.
    /// Uses spatial hash for efficient lookup — no full particle scan.
    /// </summary>
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

                // Only wake particles matching the flask target type.
                // Non-matching particles stay asleep — they'll wake via
                // cascade when absorbed neighbors start moving nearby.
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

    /// <summary>
    /// Wakes sleeping particles in a vertical column above the suction point.
    /// This prevents floating islands: when particles are removed from the middle,
    /// everything above must know there's a void below and start falling.
    /// Very cheap: only scans grid cells in a narrow vertical strip.
    /// </summary>
    void WakeColumnAbove(float2 point)
    {
        int wakeBudget = wakeBudgetPerFrame;

        // Column width: slightly wider than smoothing radius so we catch
        // particles that aren't directly above but would be affected
        float halfWidth = smoothingRadius * 1.5f;

        int2 colMin = CellCoord(new float2(point.x - halfWidth, point.y));
        int2 colMax = CellCoord(new float2(point.x + halfWidth, containerMax.y));

        // Scan upward from flask position to top of container
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

                // Only wake matching type — others wake via cascade/islands
                bool typeMatch = (flaskTargetType < 0) || (particles[i].typeIndex == flaskTargetType);
                if (!typeMatch) continue;

                float dx = particles[i].position.x - point.x;
                if (dx * dx < halfWidth * halfWidth)
                {
                    sleepState[i] = 0;
                    sleepCounter[i] = 0;
                    wakeBudget--;
                }
            }
        }
    }

    /// <summary>
    /// Detects floating islands using a grid-level flood fill from the floor.
    /// 
    /// Algorithm:
    ///   1. Mark all bottom-row grid cells that contain alive particles as "grounded"
    ///   2. Propagate upward: cell (x,y) is grounded if it has alive particles AND
    ///      any of (x-1,y-1), (x,y-1), (x+1,y-1) is grounded
    ///   3. Wake all sleeping particles in non-grounded cells
    ///
    /// Cost: O(gridW × gridH) for the flood + O(floating particles) for waking.
    /// Grid is typically ~50×50 = 2500 cells — trivially cheap every frame.
    /// </summary>
    void DetectFloatingIslands()
    {
        int wakeBudget = wakeBudgetPerFrame;

        // ── Phase 1: Flood fill "grounded" with diagonals (catches true islands) ──

        bool[] grounded = new bool[hashGridTotal];

        // Bottom row: grounded if has alive particles
        for (int x = 0; x < hashGridW; x++)
        {
            grounded[x] = cellCounts[x] > 0; // y=0 row
        }

        // Left and right wall columns are also grounded (wall support).
        // This prevents particles at container edges from being detected as floating.
        for (int y = 0; y < hashGridH; y++)
        {
            int leftCi = y * hashGridW + 0;
            int rightCi = y * hashGridW + (hashGridW - 1);
            if (cellCounts[leftCi] > 0) grounded[leftCi] = true;
            if (cellCounts[rightCi] > 0) grounded[rightCi] = true;
        }

        // Propagate upward: cell is grounded if has particles AND
        // any of (x-1,y-1), (x,y-1), (x+1,y-1) is grounded
        for (int y = 1; y < hashGridH; y++)
        {
            for (int x = 0; x < hashGridW; x++)
            {
                int ci = y * hashGridW + x;

                if (cellCounts[ci] == 0)
                {
                    grounded[ci] = false;
                    continue;
                }

                bool supported = false;
                int yBelow = y - 1;

                if (x > 0) supported |= grounded[yBelow * hashGridW + (x - 1)];
                supported |= grounded[yBelow * hashGridW + x];
                if (x < hashGridW - 1) supported |= grounded[yBelow * hashGridW + (x + 1)];

                grounded[ci] = supported;
            }
        }

        // ── Phase 2: Wake particles in true floating islands (not grounded at all) ──
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

        // ── Phase 3: Slope flow — wake grounded sleeping particles ──
        // that have NO particles directly below them.
        for (int y = 1; y < hashGridH && wakeBudget > 0; y++)
        {
            for (int x = 0; x < hashGridW && wakeBudget > 0; x++)
            {
                int ci = y * hashGridW + x;

                if (!grounded[ci]) continue;
                if (cellCounts[ci] == 0) continue;

                int belowCi = (y - 1) * hashGridW + x;
                if (cellCounts[belowCi] > 0) continue;

                WakeSleepingInCell(ci, ref wakeBudget);
            }
        }

        // ── Phase 4: Horizontal spreading — liquid doesn't form vertical walls ──
        // If a cell has empty space to its side AND the diagonal-below on that side
        // is also empty, particles should flow sideways and down.
        // This turns columns into slopes that spread out like liquid.
        for (int y = 1; y < hashGridH && wakeBudget > 0; y++)
        {
            for (int x = 0; x < hashGridW && wakeBudget > 0; x++)
            {
                int ci = y * hashGridW + x;

                if (!grounded[ci]) continue;
                if (cellCounts[ci] == 0) continue;

                // Check left side: is (x-1, y) empty AND (x-1, y-1) empty?
                // That means open space to flow into diagonally down-left.
                bool flowLeft = false;
                if (x > 0)
                {
                    bool sideEmpty = cellCounts[y * hashGridW + (x - 1)] == 0;
                    bool diagEmpty = cellCounts[(y - 1) * hashGridW + (x - 1)] == 0;
                    flowLeft = sideEmpty && diagEmpty;
                }

                // Check right side: same logic
                bool flowRight = false;
                if (x < hashGridW - 1)
                {
                    bool sideEmpty = cellCounts[y * hashGridW + (x + 1)] == 0;
                    bool diagEmpty = cellCounts[(y - 1) * hashGridW + (x + 1)] == 0;
                    flowRight = sideEmpty && diagEmpty;
                }

                if (flowLeft || flowRight)
                {
                    WakeSleepingInCell(ci, ref wakeBudget);
                }
            }
        }
    }

    /// <summary>
    /// Helper: wakes sleeping particles in a specific grid cell, respecting budget.
    /// </summary>
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
            particleMasses = particleMasses,
            densities = densities,
            pressures = pressures,
            cellSize = cellSize,
            containerMin = new float2(containerMin.x, containerMin.y),
            gridW = hashGridW, gridH = hashGridH,
            smoothingRadiusSqr = smoothingRadius * smoothingRadius,
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

        // 4. Integrate — only awake particles, includes suction
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

        // 5. Particle collision — compute corrections into separate array
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

        // Apply corrections to positions (safe — each thread writes only its own index)
        new ApplyCollisionJob
        {
            particles = particles,
            collisionCorrections = collisionCorrections,
            sleepState = sleepState
        }.Schedule(ParticleCount, 256).Complete();

        // 6. Sleep check — awake particles with low velocity go to sleep
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
            // Encode mass as density for GPU — shader can use it for particle size.
            // sqrt(mass) gives area-preserving radius scaling.
            float renderMass = particleMasses[i];
            Particles[i] = new FluidParticle
            {
                position = new Vector2(p.position.x, p.position.y),
                velocity = new Vector2(p.velocity.x, p.velocity.y),
                typeIndex = p.typeIndex,
                density = renderMass, // Shader reads this for size scaling
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

    /// <summary>
    /// Merges nearby same-type particles into larger droplets.
    /// Uses spatial hash for neighbor lookup. Runs on main thread
    /// every mergeInterval frames.
    ///
    /// When particle A absorbs particle B:
    ///   - A.mass += B.mass (up to maxMergeSize)
    ///   - A.position = weighted center of mass
    ///   - B.alive = 0 (removed from simulation)
    ///   - A inherits super-linear cohesion: cohesion *= mass^exponent
    /// </summary>
    void MergeParticles()
    {
        if (!enableMerging) return;
        if (Time.frameCount % mergeInterval != 0) return;

        float baseMergeDist = particleSpacing * 1.2f; // Slightly larger than spacing
        float mergeVelSqr = mergeVelocityThreshold * mergeVelocityThreshold;
        int merged = 0;

        for (int i = 0; i < ParticleCount; i++)
        {
            if (particles[i].alive < 0.5f) continue;
            if (sleepState[i] == 1) continue; // Only merge awake particles
            if (particleMasses[i] >= maxMergeSize) continue; // Already at max size
            if (math.lengthsq(particles[i].velocity) > mergeVelSqr) continue;

            float2 posI = particles[i].position;
            int typeI = particles[i].typeIndex;
            float radiusI = math.sqrt(particleMasses[i]);
            int2 cell = CellCoord(posI);

            // Find best merge candidate: same type, closest, also slow
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
                    if (j <= i) continue; // Avoid double-merge
                    if (particles[j].alive < 0.5f) continue;
                    if (sleepState[j] == 1) continue;
                    if (particles[j].typeIndex != typeI) continue;
                    if (math.lengthsq(particles[j].velocity) > mergeVelSqr) continue;

                    // Check if combined mass would exceed limit
                    if (particleMasses[i] + particleMasses[j] > maxMergeSize) continue;

                    // Merge distance scales with both particle sizes
                    // Two mass-1: baseDist * 1. Mass-4 + mass-1: baseDist * 1.5
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

            // Merge: A absorbs B
            float massA = particleMasses[i];
            float massB = particleMasses[bestJ];
            float totalMass = massA + massB;

            // Weighted center of mass
            var pI = particles[i];
            var pB = particles[bestJ];
            float2 newPos = (pI.position * massA + pB.position * massB) / totalMass;
            float2 newVel = (pI.velocity * massA + pB.velocity * massB) / totalMass;

            pI.position = newPos;
            pI.velocity = newVel;
            particles[i] = pI;
            particleMasses[i] = totalMass;

            // Kill absorbed particle
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
        [ReadOnly] public NativeArray<float> particleMasses;
        [NativeDisableParallelForRestriction] public NativeArray<float> densities;
        [NativeDisableParallelForRestriction] public NativeArray<float> pressures;

        public float cellSize, smoothingRadiusSqr;
        public float2 containerMin;
        public int gridW, gridH;
        public float restDensity, pressureStiffness, poly6Coeff;

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
                        // Use per-particle mass: merged particles contribute more to density
                        density += particleMasses[j] * poly6Coeff * diff * diff * diff;
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

            // Surface tension tracking: center of mass of same-type neighbors
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

                    // Pressure — all particles repel equally for stability
                    float gm = SpikyGrad(r);
                    float pressAvg = (pressures[i] + pressures[j]) * 0.5f;
                    float2 pF = dir * (-particleMass * pressAvg * gm / densityJ);

                    // Near-pressure repulsion
                    float nf = 1f - r / smoothingRadius;
                    float2 nF = dir * (nearPressureStiffness * nf * nf);

                    // Viscosity — smooths velocity differences
                    var typeJ = fluidTypeData[pJ.typeIndex];
                    float mu = (typeI.viscosity + typeJ.viscosity) * 0.5f;
                    float vl = ViscLapl(r);
                    float2 vF = mu * particleMass * (pJ.velocity - pI.velocity)
                              / densityJ * vl;

                    bool sameType = (pI.typeIndex == pJ.typeIndex);

                    // Cohesion — SAME TYPE ONLY: same-colored particles attract
                    // This creates "mercury droplet" behavior for each color
                    float2 cF = float2.zero;
                    if (sameType)
                    {
                        float t = r / smoothingRadius;
                        // Super-linear mass scaling, capped to prevent instability
                        // mass=1→1, mass=4→8, mass=10→31, mass=50→cap at maxCohesionScale
                        float massScale = math.min(
                            math.pow(particleMasses[j], cohesionMassExponent),
                            20f // Cap: even mass-50 won't exceed 20× cohesion
                        );
                        cF = -dir * typeI.cohesion * cohesionStrength * massScale * t * (1f - t) * (1f - t);

                        // Track center of mass for surface tension
                        float w = (1f - t) * math.min(particleMasses[j], 10f);
                        sameTypeCOM += pJ.position * w;
                        sameTypeWeight += w;
                    }

                    // Inter-type repulsion — DIFFERENT TYPES: pushes apart
                    // Creates immiscible boundaries between colors
                    float2 rF = float2.zero;
                    if (!sameType && interTypeRepulsion > 0f)
                    {
                        float rf = 1f - r / smoothingRadius;
                        rF = dir * interTypeRepulsion * rf * rf;
                    }

                    totalForce += pF + nF + vF + cF + rF;
                }
            }

            // Surface tension — pulls particle toward center of same-type neighbors
            // This is what makes droplets round (like mercury) instead of spreading flat
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

    // ─── Integrate (parallel, skips sleeping) ────────────────────

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

            // SPH forces divided by density (standard SPH).
            // Gravity: F=mg, a=g — mass cancels, all particles fall equally.
            // Cohesion jitter damping: only divide cohesion-type forces by sqrt(mass)
            // to prevent oscillation without killing movement.
            float2 accel = forces[i] / (density * math.sqrt(mass)) + gravity * gScale;

            p.velocity += accel * dt;

            // Gentle extra damping for heavy particles — prevents jitter
            // mass=1 → 1.0 (no damping), mass=10 → 0.96, mass=50 → 0.80
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

    // ─── Particle collision (parallel) ───────────────────────────
    // Enforces minimum distance between particles using spatial hash.
    // Unlike SPH pressure (which is soft), this is a hard position correction:
    // if two particles overlap, they are pushed apart directly.
    // Sleeping particles act as immovable walls — only awake particles move.

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
            // Softer radius scaling: pow(0.35) instead of sqrt(0.5).
            // mass=1 → 1.0, mass=4 → 1.6, mass=10 → 2.2, mass=50 → 3.7
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

                    // Collision distance scales with pow(mass, 0.35) — softer than sqrt
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

    // Apply collision corrections to particle positions (parallel, safe)
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