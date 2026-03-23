using UnityEngine;

/// <summary>
/// GPU-accelerated SPH fluid simulation.
/// Replaces FluidSimulation (CPU) — remove that component and add this one instead.
///
/// All physics runs on the GPU via compute shader. CPU only dispatches kernels
/// and does occasional readback for the debug overlay.
/// </summary>
public class FluidSimulationGPU : MonoBehaviour
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

    [Tooltip("How strongly particles resist compression")]
    public float pressureStiffness = 80f;
    public float nearPressureStiffness = 5f;

    // ─── Cohesion & Separation ───────────────────────────────────
    [Header("Cohesion & Separation")]
    [Tooltip("Global multiplier for same-type particle attraction. " +
             "Per-type cohesion values in Fluid Types scale this further.")]
    [Range(0f, 50f)]
    public float cohesionStrength = 15f;

    [Tooltip("How strongly different fluid types push apart. " +
             "Helps colored blobs stay separated.")]
    [Range(0f, 30f)]
    public float interTypeRepulsion = 8f;

    [Tooltip("Pulls surface particles inward toward their same-type cluster center. " +
             "Creates rounder blob shapes.")]
    [Range(0f, 30f)]
    public float surfaceTensionStrength = 5f;

    [Tooltip("When ON, all particles behave as one fluid regardless of color/type. " +
             "Cohesion applies to all neighbors, inter-type repulsion is disabled, " +
             "gravity is uniform. Color is purely visual. Auto-enabled for image mode.")]
    public bool uniformFluid = false;

    // ─── Physics ─────────────────────────────────────────────────
    [Header("Physics")]
    public Vector2 gravity = new Vector2(0f, -9.81f);

    [Range(0f, 1f)]
    public float boundaryDamping = 0.3f;

    [Range(0.9f, 1f)]
    [Tooltip("Per sub-step velocity damping. Lower = more drag")]
    public float velocityDamping = 0.98f;

    public float timeScale = 1f;

    [Range(1, 8)]
    public int subSteps = 3;

    public float maxSpeed = 8f;

    // ─── Fluid Types ─────────────────────────────────────────────
    [Header("Fluid Types")]
    public FluidTypeDefinition[] fluidTypes = new FluidTypeDefinition[]
    {
        new FluidTypeDefinition
        {
            name = "Heavy (Red)",
            color = new Color(0.9f, 0.2f, 0.15f),
            density = 3f,
            viscosity = 8f,
            cohesion = 1f
        },
        new FluidTypeDefinition
        {
            name = "Medium (Green)",
            color = new Color(0.2f, 0.85f, 0.3f),
            density = 2f,
            viscosity = 5f,
            cohesion = 0.8f
        },
        new FluidTypeDefinition
        {
            name = "Light (Blue)",
            color = new Color(0.2f, 0.4f, 0.95f),
            density = 1f,
            viscosity = 2f,
            cohesion = 0.6f
        },
    };

    // ─── Compute Shader ──────────────────────────────────────────
    [Header("GPU")]
    [Tooltip("Assign the FluidCompute.compute asset here")]
    public ComputeShader computeShader;

    // ─── Public Accessors ────────────────────────────────────────
    public FluidParticle[] Particles { get; private set; }
    public int ParticleCount { get; private set; }

    /// <summary>
    /// The GPU buffer containing particle data. Used by FluidRendererGPU for rendering.
    /// </summary>
    public ComputeBuffer ParticleBuffer => particleBuffer;

    // ─── GPU Buffers ─────────────────────────────────────────────
    private ComputeBuffer particleBuffer;
    private ComputeBuffer forcesBuffer;
    private ComputeBuffer cellCountBuffer;
    private ComputeBuffer cellParticlesBuffer;
    private ComputeBuffer fluidTypeBuffer;

    // ─── Kernel IDs ──────────────────────────────────────────────
    private int kernelClearGrid;
    private int kernelInsertParticles;
    private int kernelDensityPressure;
    private int kernelForces;
    private int kernelIntegrate;

    // ─── Grid Dimensions ─────────────────────────────────────────
    private int gpuGridWidth;
    private int gpuGridHeight;
    private int gpuGridTotalCells;
    private const int MAX_PER_CELL = 64; // Must match compute shader #define

    // ─── Thread Group Counts ─────────────────────────────────────
    private int particleGroups;
    private int gridGroups;
    private const int THREAD_GROUP_SIZE = 256; // Must match compute shader #define

    // ─── Debug Readback ──────────────────────────────────────────
    private int readbackInterval = 15; // Read back every N frames

    // ─── Lifecycle ───────────────────────────────────────────────

    void Awake()
    {
        if (computeShader == null)
        {
            Debug.LogError("[FluidSimGPU] No compute shader assigned! " +
                           "Drag FluidCompute.compute into the Compute Shader slot.");
            enabled = false;
            return;
        }

        // Check if ImageToFluid provides particle data
        var imageSource = GetComponent<ImageToFluid>();
        if (imageSource != null && imageSource.IsReady)
        {
            InitFromImage(imageSource);
        }
        else
        {
            SpawnParticles();
        }

        InitGPU();
    }

    void FixedUpdate()
    {
        float dt = (Time.fixedDeltaTime * timeScale) / subSteps;

        SetPerFrameUniforms(dt);

        for (int step = 0; step < subSteps; step++)
        {
            DispatchSimulationStep();
        }

        // Periodic readback for debug overlay
        if (Time.frameCount % readbackInterval == 0)
        {
            particleBuffer.GetData(Particles);
        }
    }

    void OnDestroy()
    {
        ReleaseBuffers();
    }

    // ─── Initialization ──────────────────────────────────────────

    void SpawnParticles()
    {
        ParticleCount = gridWidth * gridHeight;
        Particles = new FluidParticle[ParticleCount];

        float totalWidth = (gridWidth - 1) * particleSpacing;
        float totalHeight = (gridHeight - 1) * particleSpacing;
        float startX = -totalWidth * 0.5f;
        float topMargin = 0.5f;
        float startY = containerMax.y - topMargin - totalHeight;

        int typeCount = fluidTypes.Length;
        int rowsPerType = Mathf.Max(1, gridHeight / typeCount);

        for (int y = 0; y < gridHeight; y++)
        {
            int typeIndex = Mathf.Min(y / rowsPerType, typeCount - 1);

            for (int x = 0; x < gridWidth; x++)
            {
                int i = y * gridWidth + x;

                Vector2 jitter = new Vector2(
                    Random.Range(-0.01f, 0.01f),
                    Random.Range(-0.01f, 0.01f)
                );

                Particles[i] = new FluidParticle
                {
                    position = new Vector2(
                        startX + x * particleSpacing,
                        startY + y * particleSpacing
                    ) + jitter,
                    velocity = Vector2.zero,
                    typeIndex = typeIndex,
                    density = 0f,
                    pressure = 0f,
                    pad = 0f,
                    color = fluidTypes[typeIndex].color
                };
            }
        }

        Debug.Log($"[FluidSimGPU] Spawned {ParticleCount} particles " +
                  $"({gridWidth}x{gridHeight}), {typeCount} fluid types");
    }

    /// <summary>
    /// Initializes particles and fluid types from an ImageToFluid component.
    /// Replaces the default grid spawn with image-derived particle data.
    /// Also auto-scales SPH parameters to match the actual particle spacing,
    /// which can be very different from the default grid spacing.
    /// </summary>
    void InitFromImage(ImageToFluid source)
    {
        Particles = source.GeneratedParticles;
        ParticleCount = source.GeneratedParticleCount;
        fluidTypes = source.GeneratedFluidTypes;
        particleSpacing = source.ComputedSpacing;

        // Image mode: all types have uniform physics, color is cosmetic.
        uniformFluid = true;

        // ── Auto-scale SPH parameters to match particle spacing ──
        // The smoothing radius must be proportional to particle spacing.
        // Too large → too many neighbors, high density, slow.
        // Too small → particles don't see each other, no fluid behavior.
        float s = particleSpacing;
        smoothingRadius = s * 2.5f;
        particleRadius = s * 0.4f;

        // Near-pressure needs to be strong enough to prevent overlap even under gravity.
        nearPressureStiffness = 10f;
        pressureStiffness = 120f;

        // More sub-steps for stability with densely packed image particles
        subSteps = Mathf.Max(subSteps, 4);

        Debug.Log($"[FluidSimGPU] Initialized from image: {ParticleCount} particles, " +
                  $"{fluidTypes.Length} types, spacing={s:F4}, " +
                  $"smoothingRadius={smoothingRadius:F4}, uniformFluid=ON");
    }

    void InitGPU()
    {
        // ── Find kernel IDs ──
        kernelClearGrid = computeShader.FindKernel("ClearGrid");
        kernelInsertParticles = computeShader.FindKernel("InsertParticles");
        kernelDensityPressure = computeShader.FindKernel("ComputeDensityPressure");
        kernelForces = computeShader.FindKernel("ComputeForces");
        kernelIntegrate = computeShader.FindKernel("Integrate");

        // ── Compute grid dimensions ──
        float containerW = containerMax.x - containerMin.x;
        float containerH = containerMax.y - containerMin.y;
        gpuGridWidth = Mathf.CeilToInt(containerW / smoothingRadius) + 1;
        gpuGridHeight = Mathf.CeilToInt(containerH / smoothingRadius) + 1;
        gpuGridTotalCells = gpuGridWidth * gpuGridHeight;

        // ── Thread group counts ──
        particleGroups = Mathf.CeilToInt((float)ParticleCount / THREAD_GROUP_SIZE);
        gridGroups = Mathf.CeilToInt((float)gpuGridTotalCells / THREAD_GROUP_SIZE);

        // ── Create buffers ──
        // Particle buffer: 48 bytes per particle (must match struct layout)
        particleBuffer = new ComputeBuffer(ParticleCount, 48);
        particleBuffer.SetData(Particles);

        forcesBuffer = new ComputeBuffer(ParticleCount, 8); // float2 = 8 bytes
        cellCountBuffer = new ComputeBuffer(gpuGridTotalCells, 4); // int = 4 bytes
        cellParticlesBuffer = new ComputeBuffer(gpuGridTotalCells * MAX_PER_CELL, 4);

        // ── Fluid type buffer ──
        CreateFluidTypeBuffer();

        // ── Auto-calibrate rest density using CPU-side measurement ──
        if (autoRestDensity)
        {
            CalibrateRestDensity();
        }

        // ── Bind buffers to all kernels ──
        int[] allKernels = {
            kernelClearGrid, kernelInsertParticles,
            kernelDensityPressure, kernelForces, kernelIntegrate
        };

        foreach (int k in allKernels)
        {
            computeShader.SetBuffer(k, "particles", particleBuffer);
            computeShader.SetBuffer(k, "forces", forcesBuffer);
            computeShader.SetBuffer(k, "cellCount", cellCountBuffer);
            computeShader.SetBuffer(k, "cellParticles", cellParticlesBuffer);
            computeShader.SetBuffer(k, "fluidTypes", fluidTypeBuffer);
        }

        // ── Set constants that don't change per frame ──
        SetStaticUniforms();

        Debug.Log($"[FluidSimGPU] GPU initialized: grid {gpuGridWidth}x{gpuGridHeight} " +
                  $"({gpuGridTotalCells} cells), restDensity={restDensity:F1}, " +
                  $"particleGroups={particleGroups}");
    }

    /// <summary>
    /// Creates a GPU buffer with per-type fluid properties.
    /// Normalizes density values into gravity scale factors
    /// so that the average type has scale 1.0.
    /// </summary>
    void CreateFluidTypeBuffer()
    {
        // Compute average density for normalization
        float avgDensity = 0f;
        for (int i = 0; i < fluidTypes.Length; i++)
            avgDensity += fluidTypes[i].density;
        avgDensity /= fluidTypes.Length;

        // GPU struct: gravityScale, viscosity, cohesion, pad (16 bytes)
        float[] typeData = new float[fluidTypes.Length * 4];
        for (int i = 0; i < fluidTypes.Length; i++)
        {
            typeData[i * 4 + 0] = fluidTypes[i].density / avgDensity; // gravityScale
            typeData[i * 4 + 1] = fluidTypes[i].viscosity;
            typeData[i * 4 + 2] = fluidTypes[i].cohesion;
            typeData[i * 4 + 3] = 0f; // pad
        }

        fluidTypeBuffer = new ComputeBuffer(fluidTypes.Length, 16);
        fluidTypeBuffer.SetData(typeData);

        string report = "[FluidSimGPU] Fluid types: ";
        for (int i = 0; i < fluidTypes.Length; i++)
        {
            float gs = typeData[i * 4];
            report += $"{fluidTypes[i].name}(grav={gs:F2}, visc={fluidTypes[i].viscosity}) ";
        }
        Debug.Log(report);
    }

    /// <summary>
    /// Measures density of particles near the center of the particle field
    /// to set a matching rest density, preventing the initial explosion.
    /// Works with any particle layout (grid or image-derived).
    /// </summary>
    void CalibrateRestDensity()
    {
        float h = smoothingRadius;
        float hSqr = h * h;
        float h8 = hSqr * hSqr * hSqr * hSqr;
        float coeff = 4f / (Mathf.PI * h8);

        // Find the bounding box center of all particles
        Vector2 minPos = new Vector2(float.MaxValue, float.MaxValue);
        Vector2 maxPos = new Vector2(float.MinValue, float.MinValue);
        for (int i = 0; i < ParticleCount; i++)
        {
            Vector2 p = Particles[i].position;
            if (p.x < minPos.x) minPos.x = p.x;
            if (p.y < minPos.y) minPos.y = p.y;
            if (p.x > maxPos.x) maxPos.x = p.x;
            if (p.y > maxPos.y) maxPos.y = p.y;
        }
        Vector2 center = (minPos + maxPos) * 0.5f;
        float sampleRadius = Mathf.Min(maxPos.x - minPos.x, maxPos.y - minPos.y) * 0.25f;
        float sampleRadiusSqr = sampleRadius * sampleRadius;

        // Sample particles near the center where they have the most neighbors
        float totalDensity = 0f;
        int sampleCount = 0;

        for (int i = 0; i < ParticleCount; i++)
        {
            Vector2 toCenter = Particles[i].position - center;
            if (toCenter.sqrMagnitude > sampleRadiusSqr) continue;

            float density = 0f;

            for (int j = 0; j < ParticleCount; j++)
            {
                Vector2 diff = Particles[i].position - Particles[j].position;
                float rSqr = diff.sqrMagnitude;
                if (rSqr < hSqr)
                {
                    float d = hSqr - rSqr;
                    density += particleMass * coeff * d * d * d;
                }
            }

            totalDensity += density;
            sampleCount++;

            // Limit samples to keep init time reasonable for large particle counts
            if (sampleCount >= 200) break;
        }

        if (sampleCount > 0)
        {
            // In uniform/image mode, use exact measured density to prevent inflation.
            // In multi-type mode, use 95% to give slight expansion room.
            float multiplier = uniformFluid ? 1.0f : 0.95f;
            restDensity = (totalDensity / sampleCount) * multiplier;
        }

        Debug.Log($"[FluidSimGPU] Auto-calibrated restDensity = {restDensity:F1} " +
                  $"(sampled {sampleCount} particles)");
    }

    // ─── Per-Frame Uniform Updates ───────────────────────────────

    void SetStaticUniforms()
    {
        float h = smoothingRadius;
        float h2 = h * h;
        float h5 = h2 * h2 * h;
        float h8 = h2 * h2 * h2 * h2;

        computeShader.SetInt("particleCount", ParticleCount);
        computeShader.SetInt("gridWidth", gpuGridWidth);
        computeShader.SetInt("gridHeight", gpuGridHeight);
        computeShader.SetInt("gridTotalCells", gpuGridTotalCells);
        computeShader.SetFloat("cellSize", smoothingRadius);
        computeShader.SetFloat("smoothingRadius", smoothingRadius);
        computeShader.SetFloat("smoothingRadiusSqr", h2);
        computeShader.SetFloat("particleMass", particleMass);
        computeShader.SetFloat("particleRadius", particleRadius);
        computeShader.SetVector("containerMin", containerMin);
        computeShader.SetVector("containerMax", containerMax);

        // Pre-computed kernel coefficients
        computeShader.SetFloat("poly6Coeff", 4f / (Mathf.PI * h8));
        computeShader.SetFloat("spikyGradCoeff", -10f / (Mathf.PI * h5));
        computeShader.SetFloat("viscLaplCoeff", 40f / (Mathf.PI * h5));
    }

    void SetPerFrameUniforms(float dt)
    {
        computeShader.SetFloat("dt", dt);
        computeShader.SetFloat("restDensity", restDensity);
        computeShader.SetFloat("pressureStiffness", pressureStiffness);
        computeShader.SetFloat("nearPressureStiffness", nearPressureStiffness);
        computeShader.SetFloat("velocityDamping", velocityDamping);
        computeShader.SetFloat("maxSpeed", maxSpeed);
        computeShader.SetFloat("boundaryDamping", boundaryDamping);
        computeShader.SetVector("gravity", gravity);
        computeShader.SetFloat("cohesionStrength", cohesionStrength);
        computeShader.SetFloat("interTypeRepulsion", interTypeRepulsion);
        computeShader.SetFloat("surfaceTensionStrength", surfaceTensionStrength);
        computeShader.SetFloat("uniformFluid", uniformFluid ? 1f : 0f);
    }

    // ─── Simulation Dispatch ─────────────────────────────────────

    void DispatchSimulationStep()
    {
        // 1. Clear grid cell counts
        computeShader.Dispatch(kernelClearGrid, gridGroups, 1, 1);

        // 2. Insert particles into spatial hash grid
        computeShader.Dispatch(kernelInsertParticles, particleGroups, 1, 1);

        // 3. Compute density and pressure for each particle
        computeShader.Dispatch(kernelDensityPressure, particleGroups, 1, 1);

        // 4. Compute forces: pressure + near-pressure + viscosity
        computeShader.Dispatch(kernelForces, particleGroups, 1, 1);

        // 5. Integrate velocity and position, enforce boundaries
        computeShader.Dispatch(kernelIntegrate, particleGroups, 1, 1);
    }

    // ─── Cleanup ─────────────────────────────────────────────────

    void ReleaseBuffers()
    {
        particleBuffer?.Release();
        forcesBuffer?.Release();
        cellCountBuffer?.Release();
        cellParticlesBuffer?.Release();
        fluidTypeBuffer?.Release();
    }

    // ─── Debug ───────────────────────────────────────────────────

    void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        Vector3 center = new Vector3(
            (containerMin.x + containerMax.x) * 0.5f,
            (containerMin.y + containerMax.y) * 0.5f, 0f
        );
        Vector3 size = new Vector3(
            containerMax.x - containerMin.x,
            containerMax.y - containerMin.y, 0.01f
        );
        Gizmos.DrawWireCube(center, size);
    }
}