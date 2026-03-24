using UnityEngine;

/// <summary>
/// UI for flask color selection and absorbed particle counts.
/// </summary>
public class FlaskUI : MonoBehaviour
{
    [Header("UI Layout")]
    public int buttonWidth = 80;
    public int buttonHeight = 60;
    public int bottomMargin = 20;

    [Header("Cursor Indicator")]
    public bool showSuctionRadius = true;

    private FlaskController flask;
    private FluidTypeDefinition[] fluidTypes;
    private Material circleMaterial;
    private Texture2D[] colorTextures;
    private Texture2D selectedBorderTex;

    void Start()
    {
        flask = FindObjectOfType<FlaskController>();
        if (flask == null) { enabled = false; return; }

        var sim = FindObjectOfType<FluidSimulationJobs>();
        if (sim != null && sim.enabled)
            fluidTypes = sim.fluidTypes;

        if (fluidTypes == null || fluidTypes.Length == 0) { enabled = false; return; }

        BuildColorTextures();
        CreateCircleMaterial();
    }

    void BuildColorTextures()
    {
        colorTextures = new Texture2D[fluidTypes.Length];
        for (int i = 0; i < fluidTypes.Length; i++)
            colorTextures[i] = MakeSolidTexture(fluidTypes[i].color);
        selectedBorderTex = MakeSolidTexture(Color.white);
    }

    Texture2D MakeSolidTexture(Color color)
    {
        var tex = new Texture2D(1, 1);
        tex.SetPixel(0, 0, color);
        tex.Apply();
        return tex;
    }

    void OnGUI()
    {
        if (flask == null || fluidTypes == null) return;

        int count = fluidTypes.Length;
        float totalWidth = count * (buttonWidth + 10) - 10;
        float startX = (Screen.width - totalWidth) * 0.5f;
        float y = Screen.height - buttonHeight - bottomMargin;

        var countStyle = new GUIStyle(GUI.skin.label)
        { alignment = TextAnchor.MiddleCenter, fontSize = 14, fontStyle = FontStyle.Bold };
        countStyle.normal.textColor = Color.white;

        var shadowStyle = new GUIStyle(countStyle);
        shadowStyle.normal.textColor = new Color(0, 0, 0, 0.7f);

        for (int i = 0; i < count; i++)
        {
            float x = startX + i * (buttonWidth + 10);
            Rect btnRect = new Rect(x, y, buttonWidth, buttonHeight);

            if (flask.targetTypeIndex == i)
                GUI.DrawTexture(new Rect(x - 3, y - 3, buttonWidth + 6, buttonHeight + 6), selectedBorderTex);

            GUI.DrawTexture(btnRect, colorTextures[i]);

            int absorbed = (flask.AbsorbedCounts != null && i < flask.AbsorbedCounts.Length)
                ? flask.AbsorbedCounts[i] : 0;

            GUI.Label(new Rect(btnRect.x + 1, btnRect.y + 1, btnRect.width, btnRect.height),
                      absorbed.ToString(), shadowStyle);
            GUI.Label(btnRect, absorbed.ToString(), countStyle);

            if (GUI.Button(btnRect, GUIContent.none, GUIStyle.none))
                flask.SetTargetType(i);
        }

        var instrStyle = new GUIStyle(GUI.skin.label)
        { alignment = TextAnchor.MiddleCenter, fontSize = 13 };
        instrStyle.normal.textColor = new Color(1, 1, 1, 0.6f);

        string typeName = (flask.targetTypeIndex >= 0 && flask.targetTypeIndex < fluidTypes.Length)
            ? fluidTypes[flask.targetTypeIndex].name : "???";
        GUI.Label(new Rect(0, y - 30, Screen.width, 25),
            $"Selected: {typeName}  |  Hold LMB to suck  |  Total: {flask.TotalAbsorbed}", instrStyle);
    }

    void CreateCircleMaterial()
    {
        Shader shader = Shader.Find("Hidden/Internal-Colored");
        if (shader == null) return;
        circleMaterial = new Material(shader);
        circleMaterial.hideFlags = HideFlags.HideAndDontSave;
        circleMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        circleMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        circleMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
        circleMaterial.SetInt("_ZWrite", 0);
    }

    void OnRenderObject()
    {
        if (!showSuctionRadius || flask == null || circleMaterial == null || !flask.IsSucking) return;

        circleMaterial.SetPass(0);
        GL.PushMatrix();
        GL.MultMatrix(Matrix4x4.identity);

        Color ringColor = Color.white;
        if (flask.targetTypeIndex >= 0 && flask.targetTypeIndex < fluidTypes.Length)
            ringColor = fluidTypes[flask.targetTypeIndex].color;

        ringColor.a = 0.6f;
        DrawCircle(flask.FlaskWorldPos, flask.suctionRadius, ringColor, 48);
        ringColor.a = 0.9f;
        DrawCircle(flask.FlaskWorldPos, flask.absorbRadius, ringColor, 24);

        GL.PopMatrix();
    }

    void DrawCircle(Vector2 center, float radius, Color color, int segments)
    {
        GL.Begin(GL.LINES); GL.Color(color);
        float step = 2f * Mathf.PI / segments;
        for (int i = 0; i < segments; i++)
        {
            float a0 = i * step, a1 = (i + 1) * step;
            GL.Vertex3(center.x + Mathf.Cos(a0) * radius, center.y + Mathf.Sin(a0) * radius, 0f);
            GL.Vertex3(center.x + Mathf.Cos(a1) * radius, center.y + Mathf.Sin(a1) * radius, 0f);
        }
        GL.End();
    }

    void OnDestroy()
    {
        if (colorTextures != null) foreach (var t in colorTextures) if (t != null) DestroyImmediate(t);
        if (selectedBorderTex != null) DestroyImmediate(selectedBorderTex);
        if (circleMaterial != null) DestroyImmediate(circleMaterial);
    }
}