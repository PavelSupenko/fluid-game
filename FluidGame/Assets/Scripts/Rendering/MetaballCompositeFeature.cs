using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

/// <summary>
/// URP Renderer Feature that composites the fluid texture over the camera output.
/// Uses the MetaballFluidRenderer's FluidRT and CompositeMaterial.
///
/// SETUP:
///   1. Select your URP Renderer Asset (e.g. UniversalRenderPipelineAsset_Renderer)
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
        if (renderingData.cameraData.cameraType != CameraType.Game &&
            renderingData.cameraData.cameraType != CameraType.SceneView)
            return;

        var metaball = MetaballFluidRenderer.Instance;
        if (metaball == null || !metaball.showMetaballs) return;
        if (metaball.FluidRT == null || metaball.CompositeMaterial == null) return;

        pass.SetData(metaball.FluidRT, metaball.CompositeMaterial);
        renderer.EnqueuePass(pass);
    }

    protected override void Dispose(bool disposing) { pass?.Dispose(); }
}

public class MetaballCompositePass : ScriptableRenderPass
{
    private const string PROFILER_TAG = "FluidComposite";
    private RenderTexture fluidRT;
    private Material compositeMaterial;
    private static readonly int TempId = Shader.PropertyToID("_FluidCompositeTmp");

    public MetaballCompositePass(RenderPassEvent evt)
    {
        renderPassEvent = evt;
        profilingSampler = new ProfilingSampler(PROFILER_TAG);
    }

    public void SetData(RenderTexture fluid, Material composite)
    {
        fluidRT = fluid;
        compositeMaterial = composite;
    }

    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        if (fluidRT == null || compositeMaterial == null) return;

        var cmd = CommandBufferPool.Get(PROFILER_TAG);
        var cameraDesc = renderingData.cameraData.cameraTargetDescriptor;
        cameraDesc.depthBufferBits = 0;

        cmd.GetTemporaryRT(TempId, cameraDesc, FilterMode.Bilinear);
        

        // Copy current camera output to temp
        cmd.Blit(BuiltinRenderTextureType.CameraTarget, TempId);

        // Set the fluid texture
        cmd.SetGlobalTexture("_FluidTex", fluidRT);

        // Blit through composite shader: reads _MainTex (scene) + _FluidTex (fluid)
        cmd.Blit(TempId, BuiltinRenderTextureType.CameraTarget, compositeMaterial);

        cmd.ReleaseTemporaryRT(TempId);

        context.ExecuteCommandBuffer(cmd);
        CommandBufferPool.Release(cmd);
    }

    public void Dispose() { }
}