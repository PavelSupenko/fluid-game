using UnityEngine;

/// <summary>
/// Renders particles directly from the GPU compute buffer using
/// DrawMeshInstancedIndirect. No CPU readback — the shader reads
/// particle positions and colors from the StructuredBuffer via SV_InstanceID.
/// 
/// Replaces FluidRenderer (CPU). Remove that component and add this one.
/// </summary>
[RequireComponent(typeof(FluidSimulationGPU))]
public class FluidRendererGPU : MonoBehaviour
{
    [Header("Rendering")]
    [Tooltip("Toggle individual particle circles on/off")]
    public bool showParticles = true;

    [Tooltip("Material using the FluidSim/ParticleCircleGPU shader")]
    public Material particleMaterial;

    [Tooltip("World-space size of each particle quad")]
    public float renderScale = 0.12f;

    // ─── Internals ───────────────────────────────────────────────

    private FluidSimulationGPU sim;
    private Mesh quadMesh;
    private ComputeBuffer argsBuffer;
    private Bounds renderBounds;
    private bool argsInitialized;

    // Args for DrawMeshInstancedIndirect:
    // [0] = index count per instance (6 for a quad)
    // [1] = instance count
    // [2] = start index
    // [3] = base vertex
    // [4] = start instance
    private uint[] args = new uint[5];

    // ─── Lifecycle ───────────────────────────────────────────────

    void Start()
    {
        sim = GetComponent<FluidSimulationGPU>();
        quadMesh = CreateQuadMesh();

        // Args buffer — created now but instance count set lazily in Update
        // because FluidSimulationGPU.Start() may not have run yet
        argsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);

        // Large bounds so Unity never frustum-culls the particles
        renderBounds = new Bounds(Vector3.zero, Vector3.one * 100f);
        argsInitialized = false;

        if (particleMaterial == null)
        {
            Debug.LogError("[FluidRendererGPU] No particle material assigned! " +
                           "Create a material with the FluidSim/ParticleCircleGPU shader.");
        }
    }

    void Update()
    {
        if (!showParticles || particleMaterial == null) return;
        if (sim.ParticleBuffer == null)
        {
            // Simulation hasn't initialized yet — wait
            return;
        }

        // Lazily initialize args once simulation is ready
        if (!argsInitialized)
        {
            args[0] = (uint)quadMesh.GetIndexCount(0); // 6 indices
            args[1] = (uint)sim.ParticleCount;
            args[2] = 0;
            args[3] = 0;
            args[4] = 0;
            argsBuffer.SetData(args);
            argsInitialized = true;

            Debug.Log($"[FluidRendererGPU] Args initialized: indexCount={args[0]}, " +
                      $"instanceCount={args[1]}");
        }

        // Pass the compute buffer and render scale to the material
        particleMaterial.SetBuffer("_Particles", sim.ParticleBuffer);
        particleMaterial.SetFloat("_RenderScale", renderScale);

        // Draw all particles in one GPU call — no batching, no CPU overhead
        Graphics.DrawMeshInstancedIndirect(
            quadMesh, 0, particleMaterial,
            renderBounds, argsBuffer
        );
    }

    void OnDestroy()
    {
        argsBuffer?.Release();
    }

    // ─── Helpers ─────────────────────────────────────────────────

    Mesh CreateQuadMesh()
    {
        var mesh = new Mesh { name = "ParticleQuad" };

        mesh.vertices = new Vector3[]
        {
            new Vector3(-0.5f, -0.5f, 0f),
            new Vector3( 0.5f, -0.5f, 0f),
            new Vector3( 0.5f,  0.5f, 0f),
            new Vector3(-0.5f,  0.5f, 0f),
        };

        mesh.uv = new Vector2[]
        {
            new Vector2(0f, 0f),
            new Vector2(1f, 0f),
            new Vector2(1f, 1f),
            new Vector2(0f, 1f),
        };

        mesh.triangles = new int[] { 0, 2, 1, 0, 3, 2 };
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        return mesh;
    }
}
