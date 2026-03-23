using UnityEngine;

/// <summary>
/// Renders fluid particles as a smooth metaball surface.
/// 
/// SETUP: Attach this component to the Main Camera.
/// It finds FluidSimulationGPU automatically and renders the fluid
/// as a post-process effect using OnRenderImage.
///
/// Two-pass approach:
///   1. Splat: render particles as soft gaussian blobs to an offscreen RT
///      with additive blending. Overlapping blobs accumulate weight.
///   2. Composite: threshold the accumulated weight to create sharp fluid
///      boundaries, then overlay onto the scene.
/// </summary>
[RequireComponent(typeof(Camera))]
public class MetaballFluidRenderer : MonoBehaviour
{
    [Header("Enable / Disable")]
    [Tooltip("Toggle metaball rendering on/off")]
    public bool showMetaballs = true;

    [Header("Splat Settings")]
    [Tooltip("Size of each particle's gaussian blob. Larger = more merging.")]
    public float splatScale = 0.35f;

    [Tooltip("Controls falloff sharpness. Lower = softer blobs, more merging.")]
    [Range(1f, 8f)]
    public float blobSharpness = 3f;

    [Tooltip("Render target resolution multiplier (1 = full res, 0.5 = half)")]
    [Range(0.25f, 1f)]
    public float resolutionScale = 0.75f;

    [Header("Composite Settings")]
    [Tooltip("How much accumulated weight = solid fluid. Lower = thicker fluid.")]
    [Range(0.01f, 2f)]
    public float threshold = 0.35f;

    [Tooltip("Smoothness of the fluid edge. Higher = softer boundary.")]
    [Range(0.01f, 0.5f)]
    public float edgeSoftness = 0.08f;

    [Tooltip("Bright rim at fluid edges for a glossy look")]
    [Range(0f, 1f)]
    public float edgeHighlight = 0.25f;

    [Tooltip("Color vibrancy boost")]
    [Range(0.5f, 2f)]
    public float colorSaturation = 1.3f;

    [Header("Shaders (auto-found if left empty)")]
    public Shader splatShader;
    public Shader compositeShader;

    // ─── Internals ───────────────────────────────────────────────

    private FluidSimulationGPU sim;
    private Camera cam;

    private Material splatMaterial;
    private Material compositeMaterial;

    private Mesh quadMesh;
    private ComputeBuffer argsBuffer;
    private uint[] args = new uint[5];
    private Bounds renderBounds;
    private bool initialized;

    // ─── Lifecycle ───────────────────────────────────────────────

    void Start()
    {
        cam = GetComponent<Camera>();
        sim = FindObjectOfType<FluidSimulationGPU>();

        if (sim == null)
        {
            Debug.LogError("[MetaballRenderer] No FluidSimulationGPU found in scene!");
            enabled = false;
            return;
        }

        // Find shaders if not assigned
        if (splatShader == null)
            splatShader = Shader.Find("FluidSim/MetaballSplat");
        if (compositeShader == null)
            compositeShader = Shader.Find("FluidSim/MetaballComposite");

        if (splatShader == null || compositeShader == null)
        {
            Debug.LogError("[MetaballRenderer] Could not find metaball shaders! " +
                           "Make sure MetaballSplat.shader and MetaballComposite.shader are in the project.");
            enabled = false;
            return;
        }

        splatMaterial = new Material(splatShader) { hideFlags = HideFlags.HideAndDontSave };
        compositeMaterial = new Material(compositeShader) { hideFlags = HideFlags.HideAndDontSave };

        quadMesh = CreateQuadMesh();
        renderBounds = new Bounds(Vector3.zero, Vector3.one * 100f);

        // Args buffer — filled lazily like FluidRendererGPU
        argsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);

        Debug.Log("[MetaballRenderer] Initialized successfully");
    }

    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        if (!showMetaballs || sim == null || sim.ParticleBuffer == null)
        {
            Graphics.Blit(src, dest);
            return;
        }

        EnsureArgsBuffer();

        int rtWidth = Mathf.Max(1, (int)(src.width * resolutionScale));
        int rtHeight = Mathf.Max(1, (int)(src.height * resolutionScale));

        // ── Pass 1: Render soft blobs to offscreen RT ──
        RenderTexture splatRT = RenderTexture.GetTemporary(
            rtWidth, rtHeight, 0, RenderTextureFormat.ARGBHalf
        );
        splatRT.filterMode = FilterMode.Bilinear;

        // Compute the View-Projection matrix for the splat shader.
        // GL.GetGPUProjectionMatrix handles platform differences (Y-flip for RT rendering).
        Matrix4x4 view = cam.worldToCameraMatrix;
        Matrix4x4 proj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
        Matrix4x4 vp = proj * view;

        // Set render target and clear to transparent black
        RenderTexture prevRT = RenderTexture.active;
        Graphics.SetRenderTarget(splatRT);
        GL.Clear(true, true, Color.clear);

        // Configure splat material
        splatMaterial.SetBuffer("_Particles", sim.ParticleBuffer);
        splatMaterial.SetFloat("_RenderScale", splatScale);
        splatMaterial.SetFloat("_BlobSharpness", blobSharpness);
        splatMaterial.SetMatrix("_ViewProj", vp);

        // Draw all particles as soft blobs
        splatMaterial.SetPass(0);
        Graphics.DrawMeshInstancedIndirect(
            quadMesh, 0, splatMaterial,
            renderBounds, argsBuffer
        );

        // Restore previous render target
        RenderTexture.active = prevRT;

        // ── Pass 2: Threshold and composite onto scene ──
        compositeMaterial.SetTexture("_SplatTex", splatRT);
        compositeMaterial.SetFloat("_Threshold", threshold);
        compositeMaterial.SetFloat("_EdgeSoftness", edgeSoftness);
        compositeMaterial.SetFloat("_EdgeHighlight", edgeHighlight);
        compositeMaterial.SetFloat("_ColorSaturation", colorSaturation);

        Graphics.Blit(src, dest, compositeMaterial);

        RenderTexture.ReleaseTemporary(splatRT);
    }

    void OnDestroy()
    {
        argsBuffer?.Release();
        if (splatMaterial != null) DestroyImmediate(splatMaterial);
        if (compositeMaterial != null) DestroyImmediate(compositeMaterial);
    }

    // ─── Helpers ─────────────────────────────────────────────────

    /// <summary>
    /// Initializes the args buffer on first use, once simulation has particle count ready.
    /// </summary>
    void EnsureArgsBuffer()
    {
        if (!initialized && sim.ParticleCount > 0)
        {
            args[0] = (uint)quadMesh.GetIndexCount(0);
            args[1] = (uint)sim.ParticleCount;
            args[2] = 0;
            args[3] = 0;
            args[4] = 0;
            argsBuffer.SetData(args);
            initialized = true;

            Debug.Log($"[MetaballRenderer] Args set: {sim.ParticleCount} particles");
        }
    }

    Mesh CreateQuadMesh()
    {
        var mesh = new Mesh { name = "MetaballQuad" };

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
