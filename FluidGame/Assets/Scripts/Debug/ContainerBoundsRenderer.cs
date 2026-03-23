using UnityEngine;

/// <summary>
/// Draws the container boundary as visible lines in both Game and Scene views.
/// Uses OnRenderObject + GL.Lines which renders through the camera directly.
/// </summary>
[RequireComponent(typeof(FluidSimulation))]
public class ContainerBoundsRenderer : MonoBehaviour
{
    [Tooltip("Color of the container outline")]
    public Color lineColor = Color.yellow;

    [Tooltip("Slight inward offset so the line doesn't clip at screen edge")]
    public float inset = 0.02f;

    private FluidSimulation sim;
    private Material lineMaterial;

    void Start()
    {
        sim = GetComponent<FluidSimulation>();
        CreateLineMaterial();
    }

    /// <summary>
    /// Creates an unlit material for GL line drawing.
    /// </summary>
    void CreateLineMaterial()
    {
        // Unity built-in shader for colored lines
        Shader shader = Shader.Find("Hidden/Internal-Colored");
        lineMaterial = new Material(shader);
        lineMaterial.hideFlags = HideFlags.HideAndDontSave;

        // Enable alpha blending, disable backface culling and depth writes
        lineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        lineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
        lineMaterial.SetInt("_ZWrite", 0);
    }

    /// <summary>
    /// Draws lines after all regular rendering is done.
    /// Works in both Game view and Scene view.
    /// </summary>
    void OnRenderObject()
    {
        if (sim == null || lineMaterial == null) return;

        lineMaterial.SetPass(0);

        GL.PushMatrix();
        GL.MultMatrix(Matrix4x4.identity);

        GL.Begin(GL.LINES);
        GL.Color(lineColor);

        float x0 = sim.containerMin.x + inset;
        float y0 = sim.containerMin.y + inset;
        float x1 = sim.containerMax.x - inset;
        float y1 = sim.containerMax.y - inset;

        // Bottom edge
        GL.Vertex3(x0, y0, 0f);
        GL.Vertex3(x1, y0, 0f);

        // Right edge
        GL.Vertex3(x1, y0, 0f);
        GL.Vertex3(x1, y1, 0f);

        // Top edge
        GL.Vertex3(x1, y1, 0f);
        GL.Vertex3(x0, y1, 0f);

        // Left edge
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
