using UnityEngine;

/// <summary>
/// Core fluid simulation controller with SPH (Smoothed Particle Hydrodynamics).
/// 
/// Stage 1: gravity + boundary collisions.
/// Stage 2: spatial hashing, density estimation (Poly6), pressure forces (Spiky).
/// 
/// The simulation runs in sub-steps for numerical stability since SPH pressure
/// forces can be very stiff. Each FixedUpdate is divided into multiple smaller steps.
/// </summary>
public class FluidSimulation : MonoBehaviour
{
    // ─── Container ───────────────────────────────────────────────
    [Header("Container Bounds")]
    [Tooltip("Bottom-left corner of the simulation area")]
    public Vector2 containerMin = new Vector2(-4f, -3f);
    [Tooltip("Top-right corner of the simulation area")]
    public Vector2 containerMax = new Vector2(4f, 4f);

    // ─── Particle Spawning ───────────────────────────────────────
    [Header("Particle Grid")]
    public int gridWidth = 30;
    public int gridHeight = 20;
    [Tooltip("Distance between particles in the initial grid")]
    public float particleSpacing = 0.15f;
    [Tooltip("Visual and collision radius of each particle")]
    public float particleRadius = 0.05f;

    // ─── SPH Parameters ─────────────────────────────────────────
    [Header("SPH Settings")]
    [Tooltip("Radius of influence for SPH kernels. ~2-3x particle spacing.")]
    public float smoothingRadius = 0.4f;

    [Tooltip("Mass of each particle. Affects density and force magnitudes.")]
    public float particleMass = 1f;

    [Tooltip("Target resting density. Particles push apart when above this, attract when below.")]
    public float restDensity = 15f;

    [Tooltip("How strongly particles resist compression. Higher = stiffer, more incompressible.")]
    public float pressureStiffness = 200f;

    [Tooltip("Small near-pressure term to prevent particle clumping at very close range.")]
    public float nearPressureStiffness = 18f;

    // ─── Physics ─────────────────────────────────────────────────
    [Header("Physics")]
    public Vector2 gravity = new Vector2(0f, -9.81f);
    [Range(0f, 1f)]
    [Tooltip("Velocity multiplier on boundary bounce")]
    public float boundaryDamping = 0.5f;
    [Tooltip("Global simulation speed multiplier")]
    public float timeScale = 1f;
    [Range(1, 8)]
    [Tooltip("Number of sub-steps per FixedUpdate. More = stable but slower.")]
    public int subSteps = 4;
    [Tooltip("Caps maximum particle speed to prevent explosions")]
    public float maxSpeed = 15f;

    // ─── Fluid Types ─────────────────────────────────────────────
    [Header("Fluid Types")]
    public FluidTypeDefinition[] fluidTypes = new FluidTypeDefinition[]
    {
        new FluidTypeDefinition
        {
            name = "Heavy (Red)",
            color = new Color(0.9f, 0.2f, 0.15f),
            density = 3f,
            viscosity = 0.5f,
            cohesion = 1f
        },
        new FluidTypeDefinition
        {
            name = "Medium (Green)",
            color = new Color(0.2f, 0.85f, 0.3f),
            density = 2f,
            viscosity = 0.3f,
            cohesion = 0.8f
        },
        new FluidTypeDefinition
        {
            name = "Light (Blue)",
            color = new Color(0.2f, 0.4f, 0.95f),
            density = 1f,
            viscosity = 0.1f,
            cohesion = 0.6f
        },
    };

    // ─── Public Accessors ────────────────────────────────────────
    public FluidParticle[] Particles { get; private set; }
    public int ParticleCount { get; private set; }

    // ─── Internal State ──────────────────────────────────────────
    private SpatialHash spatialHash;
    private float poly6Coeff;
    private float spikyGradCoeff;
    private float smoothingRadiusSqr;

    // Per-particle force accumulator (avoids modifying velocity mid-computation)
    private Vector2[] forces;

    // ─── Lifecycle ───────────────────────────────────────────────

    void Start()
    {
        SpawnParticles();
        InitSPH();
    }

    void FixedUpdate()
    {
        float dt = (Time.fixedDeltaTime * timeScale) / subSteps;

        for (int step = 0; step < subSteps; step++)
        {
            SimulationStep(dt);
        }
    }

    /// <summary>
    /// A single simulation sub-step: hash → density → pressure → integrate → clamp.
    /// </summary>
    void SimulationStep(float dt)
    {
        spatialHash.Rebuild(Particles, ParticleCount);
        ComputeDensityAndPressure();
        ComputePressureForces();
        ApplyGravity(dt);
        IntegrateAndClamp(dt);
        EnforceBoundaries();
    }

    // ─── Initialization ──────────────────────────────────────────

    void InitSPH()
    {
        spatialHash = new SpatialHash(smoothingRadius, ParticleCount);
        forces = new Vector2[ParticleCount];

        // Precompute kernel coefficients (only change if smoothingRadius changes)
        poly6Coeff = SPHKernels.Poly6Coefficient(smoothingRadius);
        spikyGradCoeff = SPHKernels.SpikyGradCoefficient(smoothingRadius);
        smoothingRadiusSqr = smoothingRadius * smoothingRadius;

        Debug.Log($"[FluidSim] SPH initialized: h={smoothingRadius}, " +
                  $"restDensity={restDensity}, stiffness={pressureStiffness}, " +
                  $"subSteps={subSteps}");
    }

    // ─── Spawning ────────────────────────────────────────────────

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
                    color = fluidTypes[typeIndex].color,
                    density = 0f,
                    pressure = 0f
                };
            }
        }

        Debug.Log($"[FluidSim] Spawned {ParticleCount} particles " +
                  $"({gridWidth}x{gridHeight}), {typeCount} fluid types");
    }

    // ─── SPH: Density & Pressure ─────────────────────────────────

    /// <summary>
    /// For each particle, sum contributions from neighbors using the Poly6 kernel
    /// to estimate local density. Then compute pressure using the equation of state:
    ///   P = k * (density - restDensity)
    /// </summary>
    void ComputeDensityAndPressure()
    {
        float massSqr = particleMass * particleMass;

        for (int i = 0; i < ParticleCount; i++)
        {
            var neighbors = spatialHash.GetNeighborCandidates(Particles[i].position);
            float density = 0f;

            for (int n = 0; n < neighbors.Count; n++)
            {
                int j = neighbors[n];
                Vector2 diff = Particles[i].position - Particles[j].position;
                float rSqr = diff.sqrMagnitude;

                // Poly6 kernel contribution — includes self (i==j, rSqr==0)
                density += particleMass * SPHKernels.Poly6(rSqr, smoothingRadiusSqr, poly6Coeff);
            }

            var p = Particles[i];
            p.density = Mathf.Max(density, 0.001f); // Prevent division by zero
            p.pressure = pressureStiffness * (p.density - restDensity);
            Particles[i] = p;
        }
    }

    // ─── SPH: Pressure Forces ────────────────────────────────────

    /// <summary>
    /// Computes the symmetric pressure force between particle pairs using
    /// the Spiky kernel gradient. The symmetric form (Pi + Pj) / 2 ensures
    /// Newton's third law is satisfied, giving stable behavior.
    /// 
    /// Also adds a near-pressure repulsion term to prevent particle stacking.
    /// </summary>
    void ComputePressureForces()
    {
        // Clear forces
        for (int i = 0; i < ParticleCount; i++)
        {
            forces[i] = Vector2.zero;
        }

        for (int i = 0; i < ParticleCount; i++)
        {
            var neighbors = spatialHash.GetNeighborCandidates(Particles[i].position);

            for (int n = 0; n < neighbors.Count; n++)
            {
                int j = neighbors[n];
                if (j == i) continue;

                Vector2 diff = Particles[i].position - Particles[j].position;
                float rSqr = diff.sqrMagnitude;

                if (rSqr >= smoothingRadiusSqr || rSqr < 1e-12f) continue;

                float r = Mathf.Sqrt(rSqr);
                Vector2 dir = diff / r; // Normalized direction from j to i

                // Symmetric pressure force: -(Pi + Pj) / (2 * densityJ) * grad(W)
                float gradMag = SPHKernels.SpikyGrad(r, smoothingRadius, spikyGradCoeff);
                float pressureAvg = (Particles[i].pressure + Particles[j].pressure) * 0.5f;
                float densityJ = Particles[j].density;

                Vector2 pressureForce = dir * (-particleMass * pressureAvg * gradMag / densityJ);

                // Near-pressure: extra repulsion at very close range to prevent clumping
                float nearFactor = 1f - r / smoothingRadius;
                Vector2 nearForce = dir * (nearPressureStiffness * nearFactor * nearFactor);

                forces[i] += pressureForce + nearForce;
            }
        }
    }

    // ─── Integration ─────────────────────────────────────────────

    void ApplyGravity(float dt)
    {
        for (int i = 0; i < ParticleCount; i++)
        {
            // Acceleration = (pressure force / density) + gravity
            Vector2 pressureAccel = forces[i] / Mathf.Max(Particles[i].density, 0.001f);
            Particles[i].velocity += (pressureAccel + gravity) * dt;
        }
    }

    /// <summary>
    /// Integrates positions and clamps max speed to prevent numerical instability.
    /// </summary>
    void IntegrateAndClamp(float dt)
    {
        float maxSpeedSqr = maxSpeed * maxSpeed;

        for (int i = 0; i < ParticleCount; i++)
        {
            // Clamp velocity
            if (Particles[i].velocity.sqrMagnitude > maxSpeedSqr)
            {
                Particles[i].velocity = Particles[i].velocity.normalized * maxSpeed;
            }

            Particles[i].position += Particles[i].velocity * dt;
        }
    }

    /// <summary>
    /// Clamps particles inside the container and reflects velocity on collision.
    /// </summary>
    void EnforceBoundaries()
    {
        float r = particleRadius;

        for (int i = 0; i < ParticleCount; i++)
        {
            var p = Particles[i];

            if (p.position.x < containerMin.x + r)
            {
                p.position.x = containerMin.x + r;
                p.velocity.x *= -boundaryDamping;
            }
            else if (p.position.x > containerMax.x - r)
            {
                p.position.x = containerMax.x - r;
                p.velocity.x *= -boundaryDamping;
            }

            if (p.position.y < containerMin.y + r)
            {
                p.position.y = containerMin.y + r;
                p.velocity.y *= -boundaryDamping;
            }
            else if (p.position.y > containerMax.y - r)
            {
                p.position.y = containerMax.y - r;
                p.velocity.y *= -boundaryDamping;
            }

            Particles[i] = p;
        }
    }

    // ─── Debug Visualization ─────────────────────────────────────

    void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        Vector3 center = new Vector3(
            (containerMin.x + containerMax.x) * 0.5f,
            (containerMin.y + containerMax.y) * 0.5f,
            0f
        );
        Vector3 size = new Vector3(
            containerMax.x - containerMin.x,
            containerMax.y - containerMin.y,
            0.01f
        );
        Gizmos.DrawWireCube(center, size);
    }
}
