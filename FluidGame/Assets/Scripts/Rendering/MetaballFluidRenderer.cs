using UnityEngine;

/// <summary>
/// Renders fluid particles as smooth colored blobs using direct alpha blending.
/// 
/// APPROACH: Renders particles as soft-edge circles to an RGBA render texture
/// using standard alpha blending (SrcAlpha/OneMinusSrcAlpha). Same-color particles
/// overlap seamlessly. Then composites over the scene using a simple blit.
///
/// NO additive blending. NO weight accumulation. NO composite shader math.
/// Just colored circles that blend properly on any background.
///
/// SETUP: Put this on the Main Camera. Enable showMetaballs.
///        Add MetaballCompositeFeature to URP Renderer Asset.
/// </summary>
[RequireComponent(typeof(Camera))]
public class MetaballFluidRenderer : MonoBehaviour
{
    public static MetaballFluidRenderer Instance { get; private set; }

    [Header("Enable / Disable")]
    public bool showMetaballs = true;

    [Header("Rendering")]
    [Tooltip("Size of each particle blob relative to spacing. Larger = more overlap/merging.")]
    public float splatScale = 0.1f;

    [Tooltip("Edge softness of each particle. Lower = sharper circles, higher = softer merge.")]
    [Range(1f, 8f)]
    public float blobSharpness = 2f;

    [Tooltip("Render target resolution multiplier")]
    [Range(0.25f, 1f)]
    public float resolutionScale = 0.75f;

    [Header("Snap to Palette")]
    [Tooltip("Each pixel gets the exact palette color (no blending between types).")]
    public bool solidColors = true;

    [Header("Debug")]
    public bool showDebugRT = false;

    // ─── Public for RendererFeature ──────────────────────────────

    public RenderTexture FluidRT { get; private set; }
    public Material CompositeMaterial { get; private set; }

    // ─── Internals ───────────────────────────────────────────────

    private ComputeBuffer simParticleBuffer;
    private int simParticleCount;
    private FluidTypeDefinition[] simFluidTypes;
    private float simParticleSpacing;
    private Camera cam;
    private Material splatMaterial;
    private Mesh quadMesh;
    private ComputeBuffer argsBuffer;
    private uint[] args = new uint[5];
    private Bounds renderBounds;
    private bool argsReady;

    void OnEnable() { Instance = this; }
    void OnDisable() { if (Instance == this) Instance = null; }

    void Start()
    {
        cam = GetComponent<Camera>();

        // Disable circle renderer if present
        var circleRenderer = FindObjectOfType<FluidRendererGPU>();
        if (circleRenderer != null && circleRenderer.enabled)
            circleRenderer.showParticles = false;

        var sim = FindObjectOfType<FluidSimulationJobs>();
        if (sim != null && sim.enabled)
        {
            simParticleBuffer = sim.ParticleBuffer;
            simParticleCount = sim.ParticleCount;
            simFluidTypes = sim.fluidTypes;
            simParticleSpacing = sim.particleSpacing;

            if (simParticleSpacing > 0.001f)
                splatScale = simParticleSpacing * 1.8f;
        }
        else
        {
            Debug.LogError("[MetaballRenderer] No FluidSimulationJobs found!");
            enabled = false;
            return;
        }

        // Splat material: alpha blended colored circles
        var splatShader = Shader.Find("FluidSim/MetaballSplat");
        if (splatShader == null)
        {
            Debug.LogError("[MetaballRenderer] MetaballSplat shader not found!");
            enabled = false;
            return;
        }
        splatMaterial = new Material(splatShader) { hideFlags = HideFlags.HideAndDontSave };

        // Composite material: simple blit overlay
        var compShader = Shader.Find("FluidSim/MetaballComposite");
        if (compShader == null)
        {
            Debug.LogError("[MetaballRenderer] MetaballComposite shader not found!");
            enabled = false;
            return;
        }
        CompositeMaterial = new Material(compShader) { hideFlags = HideFlags.HideAndDontSave };

        quadMesh = CreateQuadMesh();
        renderBounds = new Bounds(Vector3.zero, Vector3.one * 200f);
        argsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);

        Debug.Log($"[MetaballRenderer] Initialized. splatScale={splatScale:F3}");
    }

    void LateUpdate()
    {
        if (!showMetaballs || simParticleBuffer == null) return;

        EnsureArgsBuffer();
        EnsureFluidRT();
        RenderFluid();
        UpdateCompositeMaterial();
    }

    void OnGUI()
    {
        if (showDebugRT && FluidRT != null)
        {
            float size = 300f;
            float aspect = (float)FluidRT.width / FluidRT.height;
            Rect rect = new Rect(10, Screen.height - size / aspect - 10, size, size / aspect);
            GUI.DrawTexture(rect, FluidRT);
            GUI.Label(new Rect(10, rect.y - 20, 300, 20),
                $"Fluid RT: {FluidRT.width}x{FluidRT.height}");
        }
    }

    void OnDestroy()
    {
        argsBuffer?.Release();
        if (FluidRT != null) { FluidRT.Release(); DestroyImmediate(FluidRT); }
        if (splatMaterial != null) DestroyImmediate(splatMaterial);
        if (CompositeMaterial != null) DestroyImmediate(CompositeMaterial);
    }

    // ─── Render fluid particles to RT ────────────────────────────

    void RenderFluid()
    {
        Matrix4x4 view = cam.worldToCameraMatrix;
        Matrix4x4 proj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
        Matrix4x4 vp = proj * view;

        splatMaterial.SetBuffer("_Particles", simParticleBuffer);
        splatMaterial.SetFloat("_RenderScale", splatScale);
        splatMaterial.SetFloat("_BlobSharpness", blobSharpness);
        splatMaterial.SetMatrix("_ViewProj", vp);

        // Render to fluid RT with CLEAR to transparent
        var prevRT = RenderTexture.active;
        RenderTexture.active = FluidRT;
        GL.Clear(true, true, new Color(0, 0, 0, 0)); // Fully transparent background

        splatMaterial.SetPass(0);
        Graphics.DrawMeshInstancedIndirect(quadMesh, 0, splatMaterial, renderBounds, argsBuffer);

        RenderTexture.active = prevRT;
    }

    void UpdateCompositeMaterial()
    {
        CompositeMaterial.SetTexture("_FluidTex", FluidRT);
        CompositeMaterial.SetFloat("_SolidColors", solidColors ? 1f : 0f);

        var types = simFluidTypes;
        int count = Mathf.Min(types.Length, 16);
        Vector4[] colors = new Vector4[16];
        for (int i = 0; i < count; i++)
            colors[i] = types[i].color;

        CompositeMaterial.SetFloat("_FluidTypeCount", (float)count);
        CompositeMaterial.SetVectorArray("_FluidTypeColors", colors);
    }

    // ─── RT Management ───────────────────────────────────────────

    void EnsureFluidRT()
    {
        int w = Mathf.Max(1, (int)(cam.pixelWidth * resolutionScale));
        int h = Mathf.Max(1, (int)(cam.pixelHeight * resolutionScale));

        if (FluidRT != null && FluidRT.width == w && FluidRT.height == h)
            return;

        if (FluidRT != null) { FluidRT.Release(); DestroyImmediate(FluidRT); }

        FluidRT = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32)
        {
            filterMode = FilterMode.Bilinear,
            name = "FluidRT"
        };
        FluidRT.Create();
        Debug.Log($"[MetaballRenderer] Created fluid RT: {w}x{h}");
    }

    void EnsureArgsBuffer()
    {
        if (argsReady) return;
        if (simParticleCount <= 0) return;

        args[0] = (uint)quadMesh.GetIndexCount(0);
        args[1] = (uint)simParticleCount;
        args[2] = 0; args[3] = 0; args[4] = 0;
        argsBuffer.SetData(args);
        argsReady = true;
    }

    Mesh CreateQuadMesh()
    {
        var mesh = new Mesh { name = "MetaballQuad" };
        mesh.vertices = new Vector3[]
        {
            new(-0.5f, -0.5f, 0), new(0.5f, -0.5f, 0),
            new(0.5f, 0.5f, 0), new(-0.5f, 0.5f, 0)
        };
        mesh.uv = new Vector2[]
        {
            new(0, 0), new(1, 0), new(1, 1), new(0, 1)
        };
        mesh.triangles = new int[] { 0, 1, 2, 0, 2, 3 };
        mesh.UploadMeshData(true);
        return mesh;
    }
}