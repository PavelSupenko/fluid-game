using UnityEngine;

/// <summary>
/// Displays simulation diagnostics: particle count, FPS, and SPH density stats.
/// Works with both FluidSimulation (CPU) and FluidSimulationGPU.
/// </summary>
public class FluidDebugOverlay : MonoBehaviour
{
    // References — whichever one is found in the scene
    private FluidSimulation cpuSim;
    private FluidSimulationGPU gpuSim;

    private float fps;
    private float fpsTimer;
    private float avgDensity;
    private float maxDensity;
    private float statsTimer;
    private bool isGPU;
    private const float STATS_INTERVAL = 0.25f;

    void Start()
    {
        cpuSim = FindObjectOfType<FluidSimulation>();
        gpuSim = FindObjectOfType<FluidSimulationGPU>();
        isGPU = gpuSim != null;
    }

    void Update()
    {
        fpsTimer += (Time.unscaledDeltaTime - fpsTimer) * 0.1f;
        fps = 1f / fpsTimer;

        statsTimer -= Time.unscaledDeltaTime;
        if (statsTimer <= 0f)
        {
            statsTimer = STATS_INTERVAL;
            ComputeDensityStats();
        }
    }

    void ComputeDensityStats()
    {
        FluidParticle[] particles = null;
        int count = 0;

        if (isGPU && gpuSim != null && gpuSim.Particles != null)
        {
            particles = gpuSim.Particles;
            count = gpuSim.ParticleCount;
        }
        else if (cpuSim != null && cpuSim.Particles != null)
        {
            particles = cpuSim.Particles;
            count = cpuSim.ParticleCount;
        }

        if (particles == null || count == 0) return;

        float sum = 0f;
        float max = 0f;

        for (int i = 0; i < count; i++)
        {
            float d = particles[i].density;
            sum += d;
            if (d > max) max = d;
        }

        avgDensity = sum / count;
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

        int count = isGPU
            ? (gpuSim != null ? gpuSim.ParticleCount : 0)
            : (cpuSim != null ? cpuSim.ParticleCount : 0);

        string mode = isGPU ? "GPU" : "CPU";

        GUILayout.BeginArea(new Rect(10, 10, 350, 160));
        GUILayout.Label($"Mode: {mode}", style);
        GUILayout.Label($"Particles: {count}", style);
        GUILayout.Label($"FPS: {fps:F0}", style);
        GUILayout.Label($"Avg Density: {avgDensity:F1}", style);
        GUILayout.Label($"Max Density: {maxDensity:F1}", style);
        GUILayout.EndArea();
    }
}
