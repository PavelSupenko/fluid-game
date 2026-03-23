using UnityEngine;

/// <summary>
/// Draws the container boundary as visible lines in both Game and Scene views.
/// Works with both FluidSimulation (CPU) and FluidSimulationGPU.
/// </summary>
public class ContainerBoundsRenderer : MonoBehaviour
{
    [Tooltip("Color of the container outline")]
    public Color lineColor = Color.yellow;

    [Tooltip("Slight inward offset so the line doesn't clip at screen edge")]
    public float inset = 0.02f;

    private Vector2 containerMin;
    private Vector2 containerMax;
    private Material lineMaterial;
    private bool initialized;

    void Start()
    {
        CreateLineMaterial();

        // Try Jobs sim, then GPU sim, then legacy CPU sim
        var jobs = GetComponent<FluidSimulationJobs>();
        if (jobs != null)
        {
            containerMin = jobs.containerMin;
            containerMax = jobs.containerMax;
            initialized = true;
            return;
        }

        var gpu = GetComponent<FluidSimulationGPU>();
        if (gpu != null)
        {
            containerMin = gpu.containerMin;
            containerMax = gpu.containerMax;
            initialized = true;
            return;
        }

        var cpu = GetComponent<FluidSimulation>();
        if (cpu != null)
        {
            containerMin = cpu.containerMin;
            containerMax = cpu.containerMax;
            initialized = true;
        }
    }

    void CreateLineMaterial()
    {
        Shader shader = Shader.Find("Hidden/Internal-Colored");
        lineMaterial = new Material(shader);
        lineMaterial.hideFlags = HideFlags.HideAndDontSave;
        lineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        lineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
        lineMaterial.SetInt("_ZWrite", 0);
    }

    void OnRenderObject()
    {
        if (!initialized || lineMaterial == null) return;

        lineMaterial.SetPass(0);

        GL.PushMatrix();
        GL.MultMatrix(Matrix4x4.identity);

        GL.Begin(GL.LINES);
        GL.Color(lineColor);

        float x0 = containerMin.x + inset;
        float y0 = containerMin.y + inset;
        float x1 = containerMax.x - inset;
        float y1 = containerMax.y - inset;

        GL.Vertex3(x0, y0, 0f);
        GL.Vertex3(x1, y0, 0f);

        GL.Vertex3(x1, y0, 0f);
        GL.Vertex3(x1, y1, 0f);

        GL.Vertex3(x1, y1, 0f);
        GL.Vertex3(x0, y1, 0f);

        GL.Vertex3(x0, y1, 0f);
        GL.Vertex3(x0, y0, 0f);

        GL.End();
        GL.PopMatrix();
    }

    void OnDestroy()
    {
        if (lineMaterial != null)
        {
            DestroyImmediate(lineMaterial);
        }
    }
}
