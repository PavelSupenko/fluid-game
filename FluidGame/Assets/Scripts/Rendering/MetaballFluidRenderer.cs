using UnityEngine;

/// <summary>
/// Renders fluid particles as a smooth metaball surface.
/// 
/// SETUP: Attach this component to the Main Camera.
///        Also add the MetaballCompositeFeature to your URP Renderer Asset.
///
/// This component handles the SPLAT pass: rendering soft gaussian blobs
/// to an offscreen RenderTexture using direct Graphics calls.
/// The composite pass (thresholding + overlay) is done by MetaballCompositeFeature.
///
/// The splat is rendered in LateUpdate using Graphics.DrawMeshInstancedIndirect
/// directly — NOT through the SRP command buffer — which avoids URP compatibility issues.
/// </summary>
[RequireComponent(typeof(Camera))]
public class MetaballFluidRenderer : MonoBehaviour
{
    // ─── Singleton for RendererFeature access ────────────────────
    public static MetaballFluidRenderer Instance { get; private set; }

    [Header("Enable / Disable")]
    public bool showMetaballs = true;

    [Header("Splat Settings")]
    [Tooltip("Size of each particle's gaussian blob. Larger = more merging.")]
    public float splatScale = 0.35f;

    [Tooltip("Falloff sharpness. Lower = softer blobs, more merging.")]
    [Range(1f, 8f)]
    public float blobSharpness = 3f;

    [Tooltip("Render target resolution multiplier")]
    [Range(0.25f, 1f)]
    public float resolutionScale = 0.75f;

    [Header("Composite Settings")]
    [Tooltip("When ON, each pixel snaps to the nearest fluid type color. " +
             "When OFF, colors blend smoothly at boundaries between types.")]
    public bool solidColors = true;

    [Tooltip("Accumulated weight threshold for solid fluid. Lower = thicker.")]
    [Range(0.01f, 2f)]
    public float threshold = 0.35f;

    [Tooltip("Smoothness of the fluid edge. Lower = sharper, more paint-like.")]
    [Range(0.01f, 0.5f)]
    public float edgeSoftness = 0.05f;

    [Tooltip("Edge shading: negative = darken edges (paint depth), positive = bright rim (neon).")]
    [Range(-1f, 1f)]
    public float edgeHighlight = -0.15f;

    [Tooltip("Color vibrancy boost.")]
    [Range(0.5f, 2f)]
    public float colorSaturation = 1.1f;

    // ─── Public for RendererFeature ──────────────────────────────

    /// <summary>
    /// The RenderTexture containing accumulated splat data.
    /// Read by MetaballCompositeFeature for the composite pass.
    /// </summary>
    public RenderTexture SplatRT { get; private set; }

    /// <summary>
    /// Composite material with current settings applied.
    /// Read by MetaballCompositeFeature.
    /// </summary>
    public Material CompositeMaterial { get; private set; }

    // ─── Internals ───────────────────────────────────────────────

    private FluidSimulationGPU sim;
    private Camera cam;
    private Material splatMaterial;
    private Mesh quadMesh;
    private ComputeBuffer argsBuffer;
    private uint[] args = new uint[5];
    private Bounds renderBounds;
    private bool argsReady;

    // ─── Lifecycle ───────────────────────────────────────────────

    void OnEnable()
    {
        Instance = this;
    }

    void OnDisable()
    {
        if (Instance == this) Instance = null;
    }

    void Start()
    {
        cam = GetComponent<Camera>();
        sim = FindObjectOfType<FluidSimulationGPU>();

        if (sim == null)
        {
            Debug.LogError("[MetaballRenderer] No FluidSimulationGPU found!");
            enabled = false;
            return;
        }

        // Create materials from shaders
        var splatShader = Shader.Find("FluidSim/MetaballSplat");
        var compositeShader = Shader.Find("FluidSim/MetaballComposite");

        if (splatShader == null || compositeShader == null)
        {
            Debug.LogError("[MetaballRenderer] Shaders not found! " +
                           "Ensure MetaballSplat.shader and MetaballComposite.shader are in the project.");
            enabled = false;
            return;
        }

        splatMaterial = new Material(splatShader) { hideFlags = HideFlags.HideAndDontSave };
        CompositeMaterial = new Material(compositeShader) { hideFlags = HideFlags.HideAndDontSave };

        quadMesh = CreateQuadMesh();
        renderBounds = new Bounds(Vector3.zero, Vector3.one * 200f);
        argsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);

        Debug.Log("[MetaballRenderer] Initialized. Make sure MetaballCompositeFeature " +
                  "is added to your URP Renderer Asset.");
    }

    void LateUpdate()
    {
        if (!showMetaballs || sim == null || sim.ParticleBuffer == null)
            return;

        EnsureArgsBuffer();
        EnsureSplatRT();
        RenderSplat();
        UpdateCompositeMaterial();
    }

    void OnDestroy()
    {
        argsBuffer?.Release();
        if (SplatRT != null)
        {
            SplatRT.Release();
            DestroyImmediate(SplatRT);
        }
        if (splatMaterial != null) DestroyImmediate(splatMaterial);
        if (CompositeMaterial != null) DestroyImmediate(CompositeMaterial);
    }

    // ─── Splat Rendering ─────────────────────────────────────────

    /// <summary>
    /// Renders all particles as soft gaussian blobs to SplatRT using direct Graphics calls.
    /// Additive blending accumulates (color * weight) in RGB and (weight) in alpha.
    /// </summary>
    void RenderSplat()
    {
        // Compute View-Projection matrix. renderIntoTexture=true handles Y-flip.
        Matrix4x4 view = cam.worldToCameraMatrix;
        Matrix4x4 proj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
        Matrix4x4 vp = proj * view;

        // Set material properties
        splatMaterial.SetBuffer("_Particles", sim.ParticleBuffer);
        splatMaterial.SetFloat("_RenderScale", splatScale);
        splatMaterial.SetFloat("_BlobSharpness", blobSharpness);
        splatMaterial.SetMatrix("_ViewProj", vp);

        // Render directly to the splat RT — bypasses all SRP command buffer issues
        var prevRT = RenderTexture.active;
        RenderTexture.active = SplatRT;
        GL.Clear(true, true, Color.clear);

        splatMaterial.SetPass(0);
        Graphics.DrawMeshInstancedIndirect(quadMesh, 0, splatMaterial, renderBounds, argsBuffer);

        RenderTexture.active = prevRT;
    }

    /// <summary>
    /// Updates composite material properties from current inspector values.
    /// The RendererFeature uses this material for the final composite pass.
    /// </summary>
    void UpdateCompositeMaterial()
    {
        CompositeMaterial.SetTexture("_SplatTex", SplatRT);
        CompositeMaterial.SetFloat("_Threshold", threshold);
        CompositeMaterial.SetFloat("_EdgeSoftness", edgeSoftness);
        CompositeMaterial.SetFloat("_EdgeHighlight", edgeHighlight);
        CompositeMaterial.SetFloat("_ColorSaturation", colorSaturation);
        CompositeMaterial.SetFloat("_SolidColors", solidColors ? 1f : 0f);

        // Pass fluid type colors to the shader for nearest-color snapping
        var types = sim.fluidTypes;
        int count = Mathf.Min(types.Length, 8); // Shader supports up to 8 types
        Vector4[] colors = new Vector4[8];
        for (int i = 0; i < count; i++)
        {
            colors[i] = types[i].color;
        }
        CompositeMaterial.SetFloat("_FluidTypeCount", (float)count);
        CompositeMaterial.SetVectorArray("_FluidTypeColors", colors);
    }

    // ─── RT Management ───────────────────────────────────────────

    void EnsureSplatRT()
    {
        int w = Mathf.Max(1, (int)(cam.pixelWidth * resolutionScale));
        int h = Mathf.Max(1, (int)(cam.pixelHeight * resolutionScale));

        if (SplatRT != null && SplatRT.width == w && SplatRT.height == h)
            return;

        // Recreate RT at new resolution
        if (SplatRT != null)
        {
            SplatRT.Release();
            DestroyImmediate(SplatRT);
        }

        SplatRT = new RenderTexture(w, h, 0, RenderTextureFormat.ARGBHalf)
        {
            filterMode = FilterMode.Bilinear,
            name = "MetaballSplatRT"
        };
        SplatRT.Create();

        Debug.Log($"[MetaballRenderer] Created splat RT: {w}x{h}");
    }

    void EnsureArgsBuffer()
    {
        if (argsReady) return;
        if (sim.ParticleCount <= 0) return;

        args[0] = (uint)quadMesh.GetIndexCount(0);
        args[1] = (uint)sim.ParticleCount;
        args[2] = 0;
        args[3] = 0;
        args[4] = 0;
        argsBuffer.SetData(args);
        argsReady = true;

        Debug.Log($"[MetaballRenderer] Args ready: {sim.ParticleCount} particles");
    }

    // ─── Helpers ─────────────────────────────────────────────────

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
