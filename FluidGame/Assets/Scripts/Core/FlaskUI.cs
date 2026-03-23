using UnityEngine;

/// <summary>
/// Simple UI for flask color selection and absorbed particle counts.
/// Uses IMGUI (OnGUI) for quick prototyping — no Canvas/EventSystem needed.
///
/// Shows a row of colored buttons at the bottom of the screen, one per fluid type.
/// The selected type is highlighted. Each button shows the absorbed count.
/// Also draws a suction radius indicator around the cursor when sucking.
///
/// SETUP: Add to any GameObject in the scene. Finds FlaskController automatically.
/// </summary>
public class FlaskUI : MonoBehaviour
{
    [Header("UI Layout")]
    public int buttonWidth = 80;
    public int buttonHeight = 60;
    public int bottomMargin = 20;

    [Header("Cursor Indicator")]
    [Tooltip("Show a circle at the cursor when sucking")]
    public bool showSuctionRadius = true;

    // ─── Internals ───────────────────────────────────────────────

    private FlaskController flask;
    private FluidSimulationGPU sim;
    private Material circleMaterial;

    // Cached style textures
    private Texture2D[] colorTextures;
    private Texture2D selectedBorderTex;

    void Start()
    {
        flask = FindObjectOfType<FlaskController>();
        sim = FindObjectOfType<FluidSimulationGPU>();

        if (flask == null || sim == null)
        {
            Debug.LogError("[FlaskUI] FlaskController or FluidSimulationGPU not found!");
            enabled = false;
            return;
        }

        BuildColorTextures();
        CreateCircleMaterial();
    }

    // ─── Color Button Textures ───────────────────────────────────

    void BuildColorTextures()
    {
        var types = sim.fluidTypes;
        colorTextures = new Texture2D[types.Length];

        for (int i = 0; i < types.Length; i++)
        {
            colorTextures[i] = MakeSolidTexture(types[i].color);
        }

        // Bright white border texture for selected button
        selectedBorderTex = MakeSolidTexture(Color.white);
    }

    Texture2D MakeSolidTexture(Color color)
    {
        var tex = new Texture2D(1, 1);
        tex.SetPixel(0, 0, color);
        tex.Apply();
        return tex;
    }

    // ─── IMGUI ───────────────────────────────────────────────────

    void OnGUI()
    {
        if (flask == null || sim == null) return;

        var types = sim.fluidTypes;
        int count = types.Length;

        // Center the button row horizontally
        float totalWidth = count * (buttonWidth + 10) - 10;
        float startX = (Screen.width - totalWidth) * 0.5f;
        float y = Screen.height - buttonHeight - bottomMargin;

        // Label style for count text
        var countStyle = new GUIStyle(GUI.skin.label)
        {
            alignment = TextAnchor.MiddleCenter,
            fontSize = 14,
            fontStyle = FontStyle.Bold
        };
        countStyle.normal.textColor = Color.white;

        // Small label for type name
        var nameStyle = new GUIStyle(GUI.skin.label)
        {
            alignment = TextAnchor.UpperCenter,
            fontSize = 10
        };
        nameStyle.normal.textColor = new Color(1, 1, 1, 0.8f);

        for (int i = 0; i < count; i++)
        {
            float x = startX + i * (buttonWidth + 10);
            Rect btnRect = new Rect(x, y, buttonWidth, buttonHeight);

            bool isSelected = (flask.targetTypeIndex == i);

            // Draw selection highlight border
            if (isSelected)
            {
                Rect borderRect = new Rect(x - 3, y - 3, buttonWidth + 6, buttonHeight + 6);
                GUI.DrawTexture(borderRect, selectedBorderTex);
            }

            // Draw colored button background
            GUI.DrawTexture(btnRect, colorTextures[i]);

            // Draw absorbed count on top
            int absorbed = (flask.AbsorbedCounts != null && i < flask.AbsorbedCounts.Length)
                ? flask.AbsorbedCounts[i] : 0;

            // Dark shadow for readability on any color
            var shadowStyle = new GUIStyle(countStyle);
            shadowStyle.normal.textColor = new Color(0, 0, 0, 0.7f);
            Rect shadowRect = new Rect(btnRect.x + 1, btnRect.y + 1, btnRect.width, btnRect.height);
            GUI.Label(shadowRect, absorbed.ToString(), shadowStyle);
            GUI.Label(btnRect, absorbed.ToString(), countStyle);

            // Invisible button on top to detect clicks
            if (GUI.Button(btnRect, GUIContent.none, GUIStyle.none))
            {
                flask.SetTargetType(i);
            }
        }

        // Instructions text
        var instructionStyle = new GUIStyle(GUI.skin.label)
        {
            alignment = TextAnchor.MiddleCenter,
            fontSize = 13
        };
        instructionStyle.normal.textColor = new Color(1, 1, 1, 0.6f);

        Rect instrRect = new Rect(0, y - 30, Screen.width, 25);
        string typeName = (flask.targetTypeIndex >= 0 && flask.targetTypeIndex < types.Length)
            ? types[flask.targetTypeIndex].name : "???";
        GUI.Label(instrRect, $"Selected: {typeName}  |  Hold LMB to suck  |  Total absorbed: {flask.TotalAbsorbed}", instructionStyle);
    }

    // ─── Cursor Indicator ────────────────────────────────────────

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
        if (!showSuctionRadius || flask == null || circleMaterial == null) return;
        if (!flask.IsSucking) return;

        circleMaterial.SetPass(0);

        GL.PushMatrix();
        GL.MultMatrix(Matrix4x4.identity);

        // Draw suction radius circle
        Color ringColor = Color.white;
        if (flask.targetTypeIndex >= 0 && flask.targetTypeIndex < sim.fluidTypes.Length)
        {
            ringColor = sim.fluidTypes[flask.targetTypeIndex].color;
        }
        ringColor.a = 0.6f;

        DrawCircle(flask.FlaskWorldPos, flask.suctionRadius, ringColor, 48);

        // Draw absorb radius (inner circle, more opaque)
        ringColor.a = 0.9f;
        DrawCircle(flask.FlaskWorldPos, flask.absorbRadius, ringColor, 24);

        GL.PopMatrix();
    }

    void DrawCircle(Vector2 center, float radius, Color color, int segments)
    {
        GL.Begin(GL.LINES);
        GL.Color(color);

        float step = 2f * Mathf.PI / segments;

        for (int i = 0; i < segments; i++)
        {
            float a0 = i * step;
            float a1 = (i + 1) * step;

            GL.Vertex3(center.x + Mathf.Cos(a0) * radius, center.y + Mathf.Sin(a0) * radius, 0f);
            GL.Vertex3(center.x + Mathf.Cos(a1) * radius, center.y + Mathf.Sin(a1) * radius, 0f);
        }

        GL.End();
    }

    void OnDestroy()
    {
        // Clean up textures
        if (colorTextures != null)
            foreach (var t in colorTextures)
                if (t != null) DestroyImmediate(t);
        if (selectedBorderTex != null) DestroyImmediate(selectedBorderTex);
        if (circleMaterial != null) DestroyImmediate(circleMaterial);
    }
}
