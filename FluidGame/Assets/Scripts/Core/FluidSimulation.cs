using UnityEngine;

/// <summary>
/// Core fluid simulation controller.
/// Stage 1: gravity + boundary collisions only.
/// SPH forces (pressure, viscosity, cohesion) will be added in subsequent stages.
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

    // ─── Physics ─────────────────────────────────────────────────
    [Header("Physics")]
    public Vector2 gravity = new Vector2(0f, -9.81f);
    [Range(0f, 1f)]
    [Tooltip("Velocity multiplier on boundary bounce (0 = full stop, 1 = full bounce)")]
    public float boundaryDamping = 0.3f;
    [Tooltip("Global simulation speed multiplier")]
    public float timeScale = 1f;

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

    // ─── Lifecycle ───────────────────────────────────────────────

    void Start()
    {
        SpawnParticles();
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime * timeScale;

        ApplyGravity(dt);
        IntegratePositions(dt);
        EnforceBoundaries();
    }

    // ─── Spawning ────────────────────────────────────────────────

    /// <summary>
    /// Creates a grid of particles, divided into horizontal bands by fluid type.
    /// The heaviest type is placed at the top so it falls through lighter ones.
    /// </summary>
    void SpawnParticles()
    {
        ParticleCount = gridWidth * gridHeight;
        Particles = new FluidParticle[ParticleCount];

        // Center the grid horizontally.
        // Place the top row near the top of the container,
        // with the grid extending downward so all particles start inside bounds.
        float totalWidth = (gridWidth - 1) * particleSpacing;
        float totalHeight = (gridHeight - 1) * particleSpacing;
        float startX = -totalWidth * 0.5f;
        float topMargin = 0.5f;
        float startY = containerMax.y - topMargin - totalHeight;

        int typeCount = fluidTypes.Length;
        int rowsPerType = Mathf.Max(1, gridHeight / typeCount);

        for (int y = 0; y < gridHeight; y++)
        {
            // Assign type by horizontal band (top band = index 0, bottom band = last)
            int typeIndex = Mathf.Min(y / rowsPerType, typeCount - 1);

            for (int x = 0; x < gridWidth; x++)
            {
                int i = y * gridWidth + x;

                // Small random jitter to break grid regularity
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
                    color = fluidTypes[typeIndex].color
                };
            }
        }

        Debug.Log($"[FluidSim] Spawned {ParticleCount} particles " +
                  $"({gridWidth}x{gridHeight}), {typeCount} fluid types");
    }

    // ─── Physics Steps ───────────────────────────────────────────

    void ApplyGravity(float dt)
    {
        for (int i = 0; i < ParticleCount; i++)
        {
            Particles[i].velocity += gravity * dt;
        }
    }

    /// <summary>
    /// Simple Euler integration: position += velocity * dt
    /// </summary>
    void IntegratePositions(float dt)
    {
        for (int i = 0; i < ParticleCount; i++)
        {
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

            // Left wall
            if (p.position.x < containerMin.x + r)
            {
                p.position.x = containerMin.x + r;
                p.velocity.x *= -boundaryDamping;
            }
            // Right wall
            else if (p.position.x > containerMax.x - r)
            {
                p.position.x = containerMax.x - r;
                p.velocity.x *= -boundaryDamping;
            }

            // Floor
            if (p.position.y < containerMin.y + r)
            {
                p.position.y = containerMin.y + r;
                p.velocity.y *= -boundaryDamping;
            }
            // Ceiling
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
        // Draw container outline
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
