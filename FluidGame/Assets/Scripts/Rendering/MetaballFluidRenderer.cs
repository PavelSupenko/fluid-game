using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Renders fluid particles as colored blobs with viscous bridge connections.
///
/// Bridges: tapered connections drawn between nearby same-type particles,
/// creating the look of thick oil paint or viscous fluid that "sticks together".
/// Built on CPU using a spatial hash, rendered as instanced trapezoids.
///
/// SETUP: Put on Main Camera. Add MetaballCompositeFeature to URP Renderer Asset.
/// </summary>
[RequireComponent(typeof(Camera))]
public class MetaballFluidRenderer : MonoBehaviour
{
    public static MetaballFluidRenderer Instance { get; private set; }

    [Header("Enable / Disable")]
    public bool showMetaballs = true;

    [Header("Particle Rendering")]
    [Tooltip("Size of each particle circle")]
    public float splatScale = 0.09f;

    public float splatScaleMultiplierFromSpacing = 1.8f;

    [Tooltip("Edge softness of each circle. Lower = sharper, higher = softer.")]
    [Range(1f, 8f)]
    public float blobSharpness = 2f;

    [Tooltip("Render target resolution multiplier")]
    [Range(0.25f, 1f)]
    public float resolutionScale = 0.75f;

    [Header("Bridge Connections")]
    [Tooltip("Enable viscous bridge connections between same-type particles")]
    public bool enableBridges = true;

    [Tooltip("How far a particle can 'see' neighbors for bridging, as multiplier of splatScale. " +
             "1.0 = only touching particles. 2.0 = bridge across one gap. 5.0 = long stretchy bridges.")]
    [Range(1f, 10f)]
    public float bridgeRadiusMultiplier = 2.5f;

    [Tooltip("Opacity of bridge connections. 1.0 = fully opaque, 0.5 = semi-transparent.")]
    [Range(0.1f, 1f)]
    public float bridgeAlpha = 0.9f;

    [Tooltip("Edge softness of bridges. Lower = hard edges, higher = feathered.")]
    [Range(0.1f, 2f)]
    public float bridgeEdgeSoftness = 0.5f;

    [Tooltip("Width of bridges relative to particle size. Higher = thicker connections, fewer gaps.")]
    [Range(1f, 5f)]
    public float bridgeWidthMultiplier = 2.5f;

    [Tooltip("How often to rebuild bridges (every N frames). 1 = every frame.")]
    [Range(1, 10)]
    public int bridgeRebuildInterval = 2;

    [Header("Snap to Palette")]
    public bool solidColors = true;

    [Header("Debug")]
    public bool showDebugRT = false;

    // ─── Public for RendererFeature ──────────────────────────────

    public RenderTexture FluidRT { get; private set; }
    public Material CompositeMaterial { get; private set; }

    // ─── Internals ───────────────────────────────────────────────

    private FluidSimulationJobs sim;
    private ComputeBuffer simParticleBuffer;
    private int simParticleCount;
    private FluidTypeDefinition[] simFluidTypes;
    private float simParticleSpacing;
    private Camera cam;
    private Material splatMaterial;
    private Material bridgeMaterial;
    private Mesh quadMesh;
    private Mesh bridgeQuadMesh;
    private ComputeBuffer argsBuffer;
    private ComputeBuffer bridgeBuffer;
    private ComputeBuffer bridgeArgsBuffer;
    private uint[] args = new uint[5];
    private uint[] bridgeArgs = new uint[5];
    private Bounds renderBounds;
    private bool argsReady;

    // Bridge building
    private struct BridgeData
    {
        public Vector2 posA;
        public Vector2 posB;
        public float radiusA;
        public float radiusB;
        public Color color;
    } // 40 bytes

    private List<BridgeData> bridgeList = new List<BridgeData>(4096);
    private int currentBridgeCount;
    private const int MAX_BRIDGES = 50000;

    // Simple spatial hash for bridge building
    private Dictionary<int, List<int>> bridgeGrid = new Dictionary<int, List<int>>(512);

    void OnEnable() { Instance = this; }
    void OnDisable() { if (Instance == this) Instance = null; }

    void Start()
    {
        cam = GetComponent<Camera>();

        var circleRenderer = FindObjectOfType<FluidRendererGPU>();
        if (circleRenderer != null && circleRenderer.enabled)
            circleRenderer.showParticles = false;

        sim = FindObjectOfType<FluidSimulationJobs>();
        if (sim != null && sim.enabled)
        {
            simParticleBuffer = sim.ParticleBuffer;
            simParticleCount = sim.ParticleCount;
            simFluidTypes = sim.fluidTypes;
            simParticleSpacing = sim.particleSpacing;

            if (simParticleSpacing > 0.001f)
                splatScale = simParticleSpacing * splatScaleMultiplierFromSpacing;
        }
        else
        {
            Debug.LogError("[MetaballRenderer] No FluidSimulationJobs found!");
            enabled = false;
            return;
        }

        // Splat shader
        var splatShader = Shader.Find("FluidSim/MetaballSplat");
        if (splatShader == null) { Debug.LogError("[MetaballRenderer] MetaballSplat shader not found!"); enabled = false; return; }
        splatMaterial = new Material(splatShader) { hideFlags = HideFlags.HideAndDontSave };

        // Bridge shader
        var bridgeShader = Shader.Find("FluidSim/FluidBridge");
        if (bridgeShader == null) { Debug.LogError("[MetaballRenderer] FluidBridge shader not found!"); enabled = false; return; }
        bridgeMaterial = new Material(bridgeShader) { hideFlags = HideFlags.HideAndDontSave };

        // Composite shader
        var compShader = Shader.Find("FluidSim/MetaballComposite");
        if (compShader == null) { Debug.LogError("[MetaballRenderer] MetaballComposite shader not found!"); enabled = false; return; }
        CompositeMaterial = new Material(compShader) { hideFlags = HideFlags.HideAndDontSave };

        quadMesh = CreateQuadMesh();
        bridgeQuadMesh = CreateBridgeQuadMesh();
        renderBounds = new Bounds(Vector3.zero, Vector3.one * 200f);
        argsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);
        bridgeArgsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);

        // Pre-allocate bridge buffer
        bridgeBuffer = new ComputeBuffer(MAX_BRIDGES, 40); // 40 bytes per BridgeData

        Debug.Log($"[MetaballRenderer] Initialized with bridges. splatScale={splatScale:F3}");
    }

    void LateUpdate()
    {
        if (!showMetaballs || simParticleBuffer == null) return;

        EnsureArgsBuffer();
        EnsureFluidRT();

        // Build bridges periodically
        if (enableBridges && Time.frameCount % bridgeRebuildInterval == 0)
            BuildBridges();

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
            GUI.Label(new Rect(10, rect.y - 20, 400, 20),
                $"Fluid RT: {FluidRT.width}x{FluidRT.height} | Bridges: {currentBridgeCount}");
        }
    }

    void OnDestroy()
    {
        argsBuffer?.Release();
        bridgeArgsBuffer?.Release();
        bridgeBuffer?.Release();
        if (FluidRT != null) { FluidRT.Release(); DestroyImmediate(FluidRT); }
        if (splatMaterial != null) DestroyImmediate(splatMaterial);
        if (bridgeMaterial != null) DestroyImmediate(bridgeMaterial);
        if (CompositeMaterial != null) DestroyImmediate(CompositeMaterial);
    }

    // ═════════════════════════════════════════════════════════════
    //  BRIDGE BUILDING (CPU, main thread)
    // ═════════════════════════════════════════════════════════════

    void BuildBridges()
    {
        var particles = sim.Particles;
        if (particles == null) return;

        bridgeList.Clear();

        float bridgeRadius = splatScale * bridgeRadiusMultiplier;

        // Cell size based on splatScale — stable when bridgeRadiusMultiplier changes.
        // Not too small (avoids expensive wide searches) but fine enough for neighbor detection.
        float cellSize = Mathf.Max(splatScale, 0.02f);

        // Build spatial hash
        bridgeGrid.Clear();
        int gridW = 1000;

        for (int i = 0; i < simParticleCount; i++)
        {
            if (particles[i].alive < 0.5f) continue;

            int cx = Mathf.FloorToInt(particles[i].position.x / cellSize);
            int cy = Mathf.FloorToInt(particles[i].position.y / cellSize);
            int key = cy * gridW + cx;

            if (!bridgeGrid.ContainsKey(key))
                bridgeGrid[key] = new List<int>(8);
            bridgeGrid[key].Add(i);
        }

        // Find pairs
        for (int i = 0; i < simParticleCount; i++)
        {
            if (particles[i].alive < 0.5f) continue;
            if (bridgeList.Count >= MAX_BRIDGES) break;

            int typeI = particles[i].typeIndex;
            Vector2 posI = particles[i].position;

            // Radius scales with mass (stored in density field by UploadToGPU)
            float massI = Mathf.Max(particles[i].density, 1f);
            float radiusI = splatScale * Mathf.Pow(massI, 0.35f) * 0.5f;

            // Search radius scales with THIS particle's size —
            // bigger particles reach further to find neighbors
            float searchRadius = bridgeRadius * Mathf.Pow(massI, 0.35f);
            float searchRadiusSqr = searchRadius * searchRadius;

            int cx = Mathf.FloorToInt(posI.x / cellSize);
            int cy = Mathf.FloorToInt(posI.y / cellSize);

            // Search wider grid area for large particles, capped for performance
            int searchCells = Mathf.Min(Mathf.CeilToInt(searchRadius / cellSize), 5);
            for (int dx = -searchCells; dx <= searchCells; dx++)
            for (int dy = -searchCells; dy <= searchCells; dy++)
            {
                int key = (cy + dy) * gridW + (cx + dx);
                if (!bridgeGrid.TryGetValue(key, out var cell)) continue;

                for (int n = 0; n < cell.Count; n++)
                {
                    int j = cell[n];
                    if (j <= i) continue;
                    if (particles[j].typeIndex != typeI) continue;
                    if (particles[j].alive < 0.5f) continue;

                    Vector2 posJ = particles[j].position;
                    float distSqr = (posI - posJ).sqrMagnitude;

                    if (distSqr < searchRadiusSqr && distSqr > 0.0001f)
                    {
                        float massJ = Mathf.Max(particles[j].density, 1f);
                        float radiusJ = splatScale * Mathf.Pow(massJ, 0.35f) * 0.5f;

                        bridgeList.Add(new BridgeData
                        {
                            posA = posI,
                            posB = posJ,
                            radiusA = radiusI * bridgeWidthMultiplier,
                            radiusB = radiusJ * bridgeWidthMultiplier,
                            color = particles[i].color
                        });

                        if (bridgeList.Count >= MAX_BRIDGES) break;
                    }
                }
                if (bridgeList.Count >= MAX_BRIDGES) break;
            }
        }

        currentBridgeCount = Mathf.Min(bridgeList.Count, MAX_BRIDGES);

        // Upload to GPU
        if (currentBridgeCount > 0)
        {
            // Resize buffer if needed
            if (bridgeBuffer.count < currentBridgeCount)
            {
                bridgeBuffer.Release();
                bridgeBuffer = new ComputeBuffer(currentBridgeCount + 1024, 40);
            }

            bridgeBuffer.SetData(bridgeList, 0, 0, currentBridgeCount);

            bridgeArgs[0] = (uint)bridgeQuadMesh.GetIndexCount(0);
            bridgeArgs[1] = (uint)currentBridgeCount;
            bridgeArgs[2] = 0; bridgeArgs[3] = 0; bridgeArgs[4] = 0;
            bridgeArgsBuffer.SetData(bridgeArgs);
        }
    }

    // ═════════════════════════════════════════════════════════════
    //  RENDERING
    // ═════════════════════════════════════════════════════════════

    void RenderFluid()
    {
        Matrix4x4 view = cam.worldToCameraMatrix;
        Matrix4x4 proj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
        Matrix4x4 vp = proj * view;

        var prevRT = RenderTexture.active;
        RenderTexture.active = FluidRT;
        GL.Clear(true, true, new Color(0, 0, 0, 0));

        // Pass 1: Draw bridges FIRST (behind particles)
        if (enableBridges && currentBridgeCount > 0)
        {
            bridgeMaterial.SetBuffer("_Bridges", bridgeBuffer);
            bridgeMaterial.SetMatrix("_ViewProj", vp);
            bridgeMaterial.SetFloat("_EdgeSoftness", bridgeEdgeSoftness);
            bridgeMaterial.SetFloat("_BridgeAlpha", bridgeAlpha);

            bridgeMaterial.SetPass(0);
            Graphics.DrawMeshInstancedIndirect(
                bridgeQuadMesh, 0, bridgeMaterial, renderBounds, bridgeArgsBuffer);
        }

        // Pass 2: Draw particles ON TOP of bridges
        splatMaterial.SetBuffer("_Particles", simParticleBuffer);
        splatMaterial.SetFloat("_RenderScale", splatScale);
        splatMaterial.SetFloat("_BlobSharpness", blobSharpness);
        splatMaterial.SetMatrix("_ViewProj", vp);

        splatMaterial.SetPass(0);
        Graphics.DrawMeshInstancedIndirect(
            quadMesh, 0, splatMaterial, renderBounds, argsBuffer);

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

    // ═════════════════════════════════════════════════════════════
    //  RT + MESH HELPERS
    // ═════════════════════════════════════════════════════════════

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

    // Standard quad for particles: centered at origin
    Mesh CreateQuadMesh()
    {
        var mesh = new Mesh { name = "ParticleQuad" };
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

    // Bridge quad: vertex.x = 0..1 (A to B end), vertex.y = -0.5..0.5 (side)
    Mesh CreateBridgeQuadMesh()
    {
        var mesh = new Mesh { name = "BridgeQuad" };
        mesh.vertices = new Vector3[]
        {
            new(0f, -0.5f, 0), new(1f, -0.5f, 0),
            new(1f,  0.5f, 0), new(0f,  0.5f, 0)
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