using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

/// <summary>
/// URP Renderer Feature for metaball fluid rendering.
/// 
/// SETUP:
///   1. Select your URP Renderer Asset (e.g. UniversalRenderPipelineAsset_Renderer)
///   2. Click "Add Renderer Feature" → MetaballRenderFeature
///   3. Add MetaballSettings component to your Main Camera
///   4. Done — metaballs render automatically
///
/// Two-pass approach:
///   Pass 1 (Splat): renders particles as soft gaussian blobs to an offscreen RT
///   Pass 2 (Composite): thresholds the accumulated weight and overlays onto the scene
/// </summary>
public class MetaballRenderFeature : ScriptableRendererFeature
{
    [System.Serializable]
    public class FeatureSettings
    {
        public RenderPassEvent renderPassEvent = RenderPassEvent.AfterRenderingTransparents;
    }

    public FeatureSettings settings = new FeatureSettings();

    private MetaballRenderPass renderPass;

    public override void Create()
    {
        renderPass = new MetaballRenderPass(settings.renderPassEvent);
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        // Only render for game/scene cameras, not preview
        if (renderingData.cameraData.cameraType == CameraType.Preview)
            return;

        renderPass.Setup(renderer);
        renderer.EnqueuePass(renderPass);
    }

    protected override void Dispose(bool disposing)
    {
        renderPass?.Dispose();
    }
}

/// <summary>
/// The actual render pass that does splat + composite.
/// Finds FluidSimulationGPU and MetaballSettings at runtime.
/// </summary>
public class MetaballRenderPass : ScriptableRenderPass
{
    private const string PROFILER_TAG = "MetaballFluid";

    // Materials created from shaders
    private Material splatMaterial;
    private Material compositeMaterial;

    // Quad mesh for instanced drawing
    private Mesh quadMesh;

    // Args buffer for DrawMeshInstancedIndirect
    private ComputeBuffer argsBuffer;
    private uint[] args = new uint[5];
    private int lastParticleCount = -1;

    // RT handles
    private int splatTexId = Shader.PropertyToID("_MetaballSplatTex");

    // Runtime references (looked up each frame)
    private ScriptableRenderer renderer;

    // Cached property IDs
    private static readonly int PropViewProj = Shader.PropertyToID("_ViewProj");
    private static readonly int PropParticles = Shader.PropertyToID("_Particles");
    private static readonly int PropRenderScale = Shader.PropertyToID("_RenderScale");
    private static readonly int PropBlobSharpness = Shader.PropertyToID("_BlobSharpness");
    private static readonly int PropSplatTex = Shader.PropertyToID("_SplatTex");
    private static readonly int PropThreshold = Shader.PropertyToID("_Threshold");
    private static readonly int PropEdgeSoftness = Shader.PropertyToID("_EdgeSoftness");
    private static readonly int PropEdgeHighlight = Shader.PropertyToID("_EdgeHighlight");
    private static readonly int PropColorSaturation = Shader.PropertyToID("_ColorSaturation");

    public MetaballRenderPass(RenderPassEvent evt)
    {
        renderPassEvent = evt;
        profilingSampler = new ProfilingSampler(PROFILER_TAG);

        // Find shaders and create materials
        var splatShader = Shader.Find("FluidSim/MetaballSplat");
        var compositeShader = Shader.Find("FluidSim/MetaballComposite");

        if (splatShader != null)
            splatMaterial = CoreUtils.CreateEngineMaterial(splatShader);
        if (compositeShader != null)
            compositeMaterial = CoreUtils.CreateEngineMaterial(compositeShader);

        quadMesh = CreateQuadMesh();
        argsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);
    }

    public void Setup(ScriptableRenderer renderer)
    {
        this.renderer = renderer;
    }

    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        // Find runtime objects
        var sim = Object.FindObjectOfType<FluidSimulationGPU>();
        var settings = renderingData.cameraData.camera.GetComponent<MetaballSettings>();

        // Bail if anything is missing or disabled
        if (sim == null || sim.ParticleBuffer == null) return;
        if (settings == null || !settings.showMetaballs) return;
        if (splatMaterial == null || compositeMaterial == null) return;

        // Update args buffer if particle count changed
        UpdateArgsBuffer(sim.ParticleCount);

        var cmd = CommandBufferPool.Get(PROFILER_TAG);
        var camera = renderingData.cameraData.camera;

        // Compute render target dimensions
        int rtWidth = Mathf.Max(1, (int)(camera.pixelWidth * settings.resolutionScale));
        int rtHeight = Mathf.Max(1, (int)(camera.pixelHeight * settings.resolutionScale));

        // ── Pass 1: Splat — render soft blobs to offscreen RT ──

        cmd.GetTemporaryRT(splatTexId, rtWidth, rtHeight, 0,
            FilterMode.Bilinear, RenderTextureFormat.ARGBHalf);

        cmd.SetRenderTarget(splatTexId);
        cmd.ClearRenderTarget(true, true, Color.clear);

        // Compute VP matrix for the splat shader
        Matrix4x4 view = camera.worldToCameraMatrix;
        Matrix4x4 proj = GL.GetGPUProjectionMatrix(camera.projectionMatrix, true);
        Matrix4x4 vp = proj * view;

        // Set splat material properties
        splatMaterial.SetBuffer(PropParticles, sim.ParticleBuffer);
        splatMaterial.SetFloat(PropRenderScale, settings.splatScale);
        splatMaterial.SetFloat(PropBlobSharpness, settings.blobSharpness);
        splatMaterial.SetMatrix(PropViewProj, vp);

        // Draw all particles as soft blobs
        cmd.DrawMeshInstancedIndirect(quadMesh, 0, splatMaterial, 0, argsBuffer);

        // ── Pass 2: Composite — threshold and overlay onto scene ──

        // Get the camera's color target
#if UNITY_2022_1_OR_NEWER
        var cameraColorTarget = renderer.cameraColorTargetHandle;
#else
        var cameraColorTarget = renderer.cameraColorTarget;
#endif

        // Set composite material properties
        compositeMaterial.SetTexture(PropSplatTex, null); // Clear first
        cmd.SetGlobalTexture(PropSplatTex, splatTexId);
        compositeMaterial.SetFloat(PropThreshold, settings.threshold);
        compositeMaterial.SetFloat(PropEdgeSoftness, settings.edgeSoftness);
        compositeMaterial.SetFloat(PropEdgeHighlight, settings.edgeHighlight);
        compositeMaterial.SetFloat(PropColorSaturation, settings.colorSaturation);

        // Blit: camera color → through composite shader → back to camera color
        // We need an intermediate RT because we can't read and write the same target
        int tempId = Shader.PropertyToID("_MetaballTemp");
        cmd.GetTemporaryRT(tempId, camera.pixelWidth, camera.pixelHeight, 0,
            FilterMode.Bilinear, RenderTextureFormat.DefaultHDR);

        cmd.Blit(cameraColorTarget, tempId);
        cmd.Blit(tempId, cameraColorTarget, compositeMaterial);

        cmd.ReleaseTemporaryRT(tempId);
        cmd.ReleaseTemporaryRT(splatTexId);

        context.ExecuteCommandBuffer(cmd);
        CommandBufferPool.Release(cmd);
    }

    /// <summary>
    /// Updates the indirect args buffer when particle count changes.
    /// </summary>
    void UpdateArgsBuffer(int particleCount)
    {
        if (particleCount == lastParticleCount) return;

        args[0] = (uint)quadMesh.GetIndexCount(0); // 6 indices
        args[1] = (uint)particleCount;
        args[2] = 0;
        args[3] = 0;
        args[4] = 0;
        argsBuffer.SetData(args);
        lastParticleCount = particleCount;
    }

    public void Dispose()
    {
        argsBuffer?.Release();
        argsBuffer = null;
        CoreUtils.Destroy(splatMaterial);
        CoreUtils.Destroy(compositeMaterial);
    }

    static Mesh CreateQuadMesh()
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
