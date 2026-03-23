using UnityEngine;

/// <summary>
/// Displays particle count and FPS in the top-left corner.
/// Attach to any GameObject in the scene.
/// </summary>
public class FluidDebugOverlay : MonoBehaviour
{
    private FluidSimulation sim;
    private float fps;
    private float fpsTimer;

    void Start()
    {
        sim = FindObjectOfType<FluidSimulation>();
    }

    void Update()
    {
        // Smooth FPS counter
        fpsTimer += (Time.unscaledDeltaTime - fpsTimer) * 0.1f;
        fps = 1f / fpsTimer;
    }

    void OnGUI()
    {
        var style = new GUIStyle(GUI.skin.label)
        {
            fontSize = 16,
            fontStyle = FontStyle.Bold
        };
        style.normal.textColor = Color.white;

        int count = sim != null ? sim.ParticleCount : 0;

        GUILayout.BeginArea(new Rect(10, 10, 300, 100));
        GUILayout.Label($"Particles: {count}", style);
        GUILayout.Label($"FPS: {fps:F0}", style);
        GUILayout.EndArea();
    }
}
