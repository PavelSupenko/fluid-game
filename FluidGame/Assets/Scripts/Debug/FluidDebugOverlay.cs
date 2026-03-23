using UnityEngine;

/// <summary>
/// Displays simulation diagnostics: particle count, FPS, and SPH density stats.
/// Attach to any GameObject in the scene.
/// </summary>
public class FluidDebugOverlay : MonoBehaviour
{
    private FluidSimulation sim;
    private float fps;
    private float fpsTimer;

    // Density stats (updated less frequently to reduce overhead)
    private float avgDensity;
    private float maxDensity;
    private float statsTimer;
    private const float STATS_INTERVAL = 0.25f; // Update stats 4x per second

    void Start()
    {
        sim = FindObjectOfType<FluidSimulation>();
    }

    void Update()
    {
        // Smooth FPS counter
        fpsTimer += (Time.unscaledDeltaTime - fpsTimer) * 0.1f;
        fps = 1f / fpsTimer;

        // Periodically compute density statistics
        statsTimer -= Time.unscaledDeltaTime;
        if (statsTimer <= 0f && sim != null && sim.Particles != null)
        {
            statsTimer = STATS_INTERVAL;
            ComputeDensityStats();
        }
    }

    void ComputeDensityStats()
    {
        float sum = 0f;
        float max = 0f;

        for (int i = 0; i < sim.ParticleCount; i++)
        {
            float d = sim.Particles[i].density;
            sum += d;
            if (d > max) max = d;
        }

        avgDensity = sum / Mathf.Max(1, sim.ParticleCount);
        maxDensity = max;
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

        GUILayout.BeginArea(new Rect(10, 10, 350, 140));
        GUILayout.Label($"Particles: {count}", style);
        GUILayout.Label($"FPS: {fps:F0}", style);
        GUILayout.Label($"Avg Density: {avgDensity:F1}", style);
        GUILayout.Label($"Max Density: {maxDensity:F1}", style);
        GUILayout.EndArea();
    }
}
