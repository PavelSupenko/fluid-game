using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

/// <summary>
/// URP Renderer Feature for the metaball composite pass.
///
/// This feature does ONE thing: blits the pre-rendered splat texture
/// over the camera output using the composite shader. The heavy lifting
/// (rendering particle blobs to SplatRT) is done by MetaballFluidRenderer
/// using direct Graphics calls, so this feature is intentionally minimal.
///
/// SETUP:
///   1. Select your URP Renderer Asset
///   2. Click "Add Renderer Feature" → MetaballCompositeFeature
///   3. Make sure MetaballFluidRenderer is on your Main Camera
/// </summary>
public class MetaballCompositeFeature : ScriptableRendererFeature
{
    public RenderPassEvent passEvent = RenderPassEvent.AfterRenderingPostProcessing;

    private MetaballCompositePass pass;

    public override void Create()
    {
        pass = new MetaballCompositePass(passEvent);
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        // Skip preview cameras and overlays
        if (renderingData.cameraData.cameraType != CameraType.Game &&
            renderingData.cameraData.cameraType != CameraType.SceneView)
            return;

        // Only enqueue if the metaball renderer exists and has data
        var metaball = MetaballFluidRenderer.Instance;
        if (metaball == null || !metaball.showMetaballs) return;
        if (metaball.SplatRT == null || metaball.CompositeMaterial == null) return;

        pass.SetData(metaball.SplatRT, metaball.CompositeMaterial);
        renderer.EnqueuePass(pass);
    }

    protected override void Dispose(bool disposing)
    {
        pass?.Dispose();
    }
}

/// <summary>
/// Simple render pass that composites the metaball splat texture onto the camera output.
/// Only uses cmd.Blit — no complex draw calls.
/// </summary>
public class MetaballCompositePass : ScriptableRenderPass
{
    private const string PROFILER_TAG = "MetaballComposite";

    private RenderTexture splatRT;
    private Material compositeMaterial;

    private static readonly int TempTargetId = Shader.PropertyToID("_MetaballTempTarget");

    public MetaballCompositePass(RenderPassEvent evt)
    {
        renderPassEvent = evt;
        profilingSampler = new ProfilingSampler(PROFILER_TAG);
    }

    public void SetData(RenderTexture splat, Material composite)
    {
        splatRT = splat;
        compositeMaterial = composite;
    }

    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        if (splatRT == null || compositeMaterial == null) return;

        var cmd = CommandBufferPool.Get(PROFILER_TAG);

        // Get the camera's current color target
        // cmd.Blit works with RenderTargetIdentifier, and both RTHandle and
        // RenderTargetIdentifier are accepted — this works across URP versions.
        var cameraDesc = renderingData.cameraData.cameraTargetDescriptor;
        cameraDesc.depthBufferBits = 0;

        // Create a temporary RT to copy the current scene into
        cmd.GetTemporaryRT(TempTargetId, cameraDesc, FilterMode.Bilinear);

        // Step 1: Copy the current camera output to the temp RT
        // In URP, the source of Blit (first param) reads from the active camera target
        // when using BuiltinRenderTextureType.CameraTarget
        cmd.Blit(BuiltinRenderTextureType.CameraTarget, TempTargetId);

        // Step 2: Set the splat texture as a global so the composite shader can access it
        cmd.SetGlobalTexture("_SplatTex", splatRT);

        // Step 3: Blit through the composite shader back to the camera target
        // The composite shader reads _MainTex (= temp, i.e. the scene) and
        // _SplatTex (= fluid blobs) and combines them.
        cmd.Blit(TempTargetId, BuiltinRenderTextureType.CameraTarget, compositeMaterial);

        cmd.ReleaseTemporaryRT(TempTargetId);

        context.ExecuteCommandBuffer(cmd);
        CommandBufferPool.Release(cmd);
    }

    public void Dispose()
    {
        // Nothing to dispose — we don't own the materials or textures
    }
}
