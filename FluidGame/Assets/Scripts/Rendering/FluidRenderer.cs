using UnityEngine;

/// <summary>
/// Renders fluid particles as colored circles using GPU instancing.
/// Uses Graphics.DrawMeshInstanced with a MaterialPropertyBlock
/// to pass per-instance colors efficiently.
/// </summary>
public class FluidRenderer : MonoBehaviour
{
    [Header("Rendering")]
    [Tooltip("Material using the FluidSim/ParticleCircle shader")]
    public Material particleMaterial;

    [Tooltip("World-space size of each particle quad")]
    public float renderScale = 0.12f;

    // ─── Internals ───────────────────────────────────────────────

    private FluidSimulationJobs sim;
    private Mesh quadMesh;

    // Pre-allocated arrays to avoid GC allocations every frame
    private Matrix4x4[] batchMatrices;
    private Vector4[] batchColors;
    private MaterialPropertyBlock mpb;
    private int colorPropertyId;

    // DrawMeshInstanced hard limit per call
    private const int BATCH_SIZE = 1023;

    // ─── Lifecycle ───────────────────────────────────────────────

    void Start()
    {
        sim = GetComponent<FluidSimulationJobs>();
        quadMesh = CreateQuadMesh();

        batchMatrices = new Matrix4x4[BATCH_SIZE];
        batchColors = new Vector4[BATCH_SIZE];
        mpb = new MaterialPropertyBlock();

        // Cache the property ID — using _ParticleColor to avoid
        // conflict with the built-in _Color property on materials
        colorPropertyId = Shader.PropertyToID("_ParticleColor");

        if (particleMaterial == null)
        {
            Debug.LogError("[FluidRenderer] No particle material assigned! " +
                           "Create a material with the FluidSim/ParticleCircle shader.");
        }

        // Log particle type distribution for debugging
        LogTypeDistribution();
    }

    void LateUpdate()
    {
        if (sim.Particles == null || particleMaterial == null) return;

        int remaining = sim.ParticleCount;
        int offset = 0;

        while (remaining > 0)
        {
            int count = Mathf.Min(remaining, BATCH_SIZE);
            FillBatch(offset, count);

            mpb.SetVectorArray(colorPropertyId, batchColors);
            Graphics.DrawMeshInstanced(
                quadMesh, 0, particleMaterial,
                batchMatrices, count, mpb
            );

            offset += count;
            remaining -= count;
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────

    /// <summary>
    /// Logs how many particles belong to each fluid type.
    /// </summary>
    void LogTypeDistribution()
    {
        if (sim.Particles == null) return;

        var counts = new int[sim.fluidTypes.Length];
        for (int i = 0; i < sim.ParticleCount; i++)
        {
            int t = sim.Particles[i].typeIndex;
            if (t >= 0 && t < counts.Length) counts[t]++;
        }

        string report = "[FluidRenderer] Particle type distribution: ";
        for (int i = 0; i < counts.Length; i++)
        {
            report += $"{sim.fluidTypes[i].name}={counts[i]}  ";
        }
        Debug.Log(report);
    }

    /// <summary>
    /// Fills the pre-allocated batch arrays with transform matrices and colors.
    /// </summary>
    void FillBatch(int offset, int count)
    {
        for (int i = 0; i < count; i++)
        {
            var p = sim.Particles[offset + i];

            batchMatrices[i] = Matrix4x4.TRS(
                new Vector3(p.position.x, p.position.y, 0f),
                Quaternion.identity,
                Vector3.one * renderScale
            );

            batchColors[i] = p.color;
        }
    }

    /// <summary>
    /// Creates a simple unit quad mesh centered at origin.
    /// </summary>
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
