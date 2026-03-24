using UnityEngine;

/// <summary>
/// Displays simulation diagnostics: particle count, FPS, and SPH density stats.
/// Works with FluidSimulation (legacy CPU), FluidSimulationGPU, and FluidSimulationJobs.
/// </summary>
public class FluidDebugOverlay : MonoBehaviour
{
    private FluidParticle[] particles;
    private int particleCount;
    private string modeName = "None";

    private float fps;
    private float fpsTimer;
    private float avgDensity;
    private float maxDensity;
    private float statsTimer;
    private const float STATS_INTERVAL = 0.25f;

    // Cached references
    private FluidSimulationGPU gpuSim;
    private FluidSimulationJobs jobsSim;
    private FluidSimulation cpuSim;

    void Start()
    {
        gpuSim = FindObjectOfType<FluidSimulationGPU>();
        jobsSim = FindObjectOfType<FluidSimulationJobs>();
        cpuSim = FindObjectOfType<FluidSimulation>();
    }

    void Update()
    {
        fpsTimer += (Time.unscaledDeltaTime - fpsTimer) * 0.1f;
        fps = 1f / fpsTimer;

        statsTimer -= Time.unscaledDeltaTime;
        if (statsTimer <= 0f)
        {
            statsTimer = STATS_INTERVAL;
            RefreshParticleRef();
            ComputeDensityStats();
        }
    }

    void RefreshParticleRef()
    {
        if (jobsSim != null && jobsSim.enabled && jobsSim.Particles != null)
        {
            particles = jobsSim.Particles;
            particleCount = jobsSim.ParticleCount;
            modeName = "Jobs+Burst";
        }
        else if (gpuSim != null && gpuSim.enabled && gpuSim.Particles != null)
        {
            particles = gpuSim.Particles;
            particleCount = gpuSim.ParticleCount;
            modeName = "GPU Compute";
        }
        else if (cpuSim != null && cpuSim.enabled && cpuSim.Particles != null)
        {
            particles = cpuSim.Particles;
            particleCount = cpuSim.ParticleCount;
            modeName = "CPU (legacy)";
        }
    }

    void ComputeDensityStats()
    {
        if (particles == null || particleCount == 0) return;

        float sum = 0f;
        float max = 0f;

        for (int i = 0; i < particleCount; i++)
        {
            float d = particles[i].density;
            sum += d;
            if (d > max) max = d;
        }

        avgDensity = sum / particleCount;
        maxDensity = max;
    }

    void OnGUI()
    {
        var style = new GUIStyle(GUI.skin.label)
        {
            fontSize = 16,
            fontStyle = FontStyle.Bold
        };
        style.normal.textColor = Color.black;

        GUILayout.BeginArea(new Rect(10, 150, 350, 180));
        GUILayout.Label($"Mode: {modeName}", style);
        GUILayout.Label($"Particles: {particleCount}", style);

        // Show awake count for Jobs sim
        if (jobsSim != null && jobsSim.enabled)
            GUILayout.Label($"Awake: {jobsSim.AwakeCount} ({(particleCount > 0 ? jobsSim.AwakeCount * 100 / particleCount : 0)}%)", style);

        GUILayout.Label($"FPS: {fps:F0}", style);
        GUILayout.Label($"Avg Density: {avgDensity:F1}", style);
        GUILayout.Label($"Max Density: {maxDensity:F1}", style);
        GUILayout.EndArea();
    }
}
