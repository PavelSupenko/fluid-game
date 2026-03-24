using UnityEngine;

/// <summary>
/// Displays simulation diagnostics: particle count, FPS, awake count, soft body info.
/// </summary>
public class FluidDebugOverlay : MonoBehaviour
{
    private FluidSimulationJobs sim;
    private float fps, fpsTimer;

    void Start()
    {
        sim = FindObjectOfType<FluidSimulationJobs>();
    }

    void Update()
    {
        fpsTimer += (Time.unscaledDeltaTime - fpsTimer) * 0.1f;
        fps = 1f / fpsTimer;
    }

    void OnGUI()
    {
        var style = new GUIStyle(GUI.skin.label)
            { fontSize = 16, fontStyle = FontStyle.Bold };
        style.normal.textColor = Color.white;

        int count = sim != null ? sim.ParticleCount : 0;

        GUILayout.BeginArea(new Rect(10, 10, 400, 200));
        GUILayout.Label($"Mode: PBD Soft Body (Jobs+Burst)", style);
        GUILayout.Label($"Particles: {count}", style);

        if (sim != null && sim.enabled)
        {
            GUILayout.Label($"Awake: {sim.AwakeCount} ({(count > 0 ? sim.AwakeCount * 100 / count : 0)}%)", style);
            if (sim.HasSoftBodies)
                GUILayout.Label($"Bodies: {sim.BodyCount}  Springs: {sim.SpringCount}", style);
        }

        GUILayout.Label($"FPS: {fps:F0}", style);
        GUILayout.EndArea();
    }
}