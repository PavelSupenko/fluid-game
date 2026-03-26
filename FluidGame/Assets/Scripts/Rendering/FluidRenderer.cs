using UnityEngine;

/// <summary>
/// Renders particles from ComputeBuffer using DrawMeshInstancedIndirect.
/// Works with both FluidSimulationGPU and FluidSimulationJobs.
/// </summary>
public class FluidRenderer : MonoBehaviour
{
    [Header("Rendering")]
    [Tooltip("Toggle individual particle circles on/off")]
    public bool showParticles = true;

    [Tooltip("Material using the FluidSim/ParticleCircleGPU shader")]
    public Material particleMaterial;

    [Tooltip("World-space size of each particle quad")]
    public float renderScale = 0.12f;

    // ─── Internals ───────────────────────────────────────────────

    private ComputeBuffer particleBufferRef;
    private int particleCount;
    private Mesh quadMesh;
    private ComputeBuffer argsBuffer;
    private Bounds renderBounds;
    private bool argsInitialized;

    // Args for DrawMeshInstancedIndirect:
    private uint[] args = new uint[5];

    // ─── Lifecycle ───────────────────────────────────────────────

    void Start()
    {
        quadMesh = CreateQuadMesh();
        argsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);
        renderBounds = new Bounds(Vector3.zero, Vector3.one * 100f);
        argsInitialized = false;

        if (particleMaterial == null)
            Debug.LogError("[FluidRendererGPU] No particle material assigned!");
    }

    void Update()
    {
        if (!showParticles || particleMaterial == null) return;

        // Find particle buffer from whichever sim is active
        if (particleBufferRef == null)
        {
            var jobs = FindObjectOfType<FluidSimulationJobs>();
            if (jobs != null && jobs.enabled && jobs.ParticleBuffer != null)
            { particleBufferRef = jobs.ParticleBuffer; particleCount = jobs.ParticleCount; }

            if (particleBufferRef == null) return;
        }

        if (!argsInitialized)
        {
            args[0] = (uint)quadMesh.GetIndexCount(0);
            args[1] = (uint)particleCount;
            args[2] = 0; args[3] = 0; args[4] = 0;
            argsBuffer.SetData(args);
            argsInitialized = true;
        }

        // Pass the compute buffer and render scale to the material
        particleMaterial.SetBuffer("_Particles", particleBufferRef);
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
