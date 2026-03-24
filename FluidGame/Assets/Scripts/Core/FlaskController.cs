using UnityEngine;

/// <summary>
/// Flask controller for sucking up fluid particles.
/// Works with both FluidSimulationGPU and FluidSimulationJobs.
///
/// SETUP: Add to any GameObject. Finds the active simulation automatically.
/// </summary>
public class FlaskController : MonoBehaviour
{
    [Header("Suction Settings")]
    public float suctionRadius = 1.0f;
    public float absorbRadius = 0.15f;
    public float suctionStrength = 80f;

    [Header("Capacity")]
    public bool limitCapacity = false;
    public int maxCapacity = 500;

    [Header("Target")]
    public int targetTypeIndex = 0;

    // ─── Public State ────────────────────────────────────────────
    public int[] AbsorbedCounts { get; private set; }
    public int TotalAbsorbed { get; private set; }
    public bool IsSucking { get; private set; }
    public Vector2 FlaskWorldPos { get; private set; }

    // ─── Internals ───────────────────────────────────────────────
    private FluidSimulationJobs jobsSim;
    private Camera mainCam;
    private FluidTypeDefinition[] fluidTypes;
    private FluidParticle[] particles;
    private int particleCount;
    private int lastCountFrame = -1;

    void Start()
    {
        mainCam = Camera.main;

        jobsSim = FindObjectOfType<FluidSimulationJobs>();

        if (jobsSim != null && jobsSim.enabled)
        {
            fluidTypes = jobsSim.fluidTypes;
            particleCount = jobsSim.ParticleCount;
        }
        else
        {
            Debug.LogError("[FlaskController] No active simulation found!");
            enabled = false;
            return;
        }

        AbsorbedCounts = new int[Mathf.Max(fluidTypes.Length, 1)];
    }

    void Update()
    {
        UpdateFlaskPosition();
        IsSucking = Input.GetMouseButton(0);

        // Pass suction data to whichever sim is active
        if (jobsSim != null && jobsSim.enabled)
            PassToJobs();

        // Count absorbed periodically
        if (Time.frameCount % 15 == 0 && Time.frameCount != lastCountFrame)
        {
            CountAbsorbed();
            lastCountFrame = Time.frameCount;
        }
    }

    void UpdateFlaskPosition()
    {
        Vector3 mouseScreen = Input.mousePosition;
        mouseScreen.z = Mathf.Abs(mainCam.transform.position.z);
        Vector3 worldPos = mainCam.ScreenToWorldPoint(mouseScreen);
        FlaskWorldPos = new Vector2(worldPos.x, worldPos.y);
    }

    void PassToJobs()
    {
        jobsSim.flaskActive = IsSucking;
        jobsSim.flaskPos = new Unity.Mathematics.float2(FlaskWorldPos.x, FlaskWorldPos.y);
        jobsSim.flaskTargetType = targetTypeIndex;
        jobsSim.flaskRadius = suctionRadius;
        jobsSim.flaskAbsorbRadius = absorbRadius;
        jobsSim.flaskStrength = suctionStrength;
    }

    void CountAbsorbed()
    {
        FluidParticle[] p = null;
        int count = 0;

        if (jobsSim != null && jobsSim.enabled && jobsSim.Particles != null)
        { p = jobsSim.Particles; count = jobsSim.ParticleCount; }

        if (p == null) return;

        for (int i = 0; i < AbsorbedCounts.Length; i++) AbsorbedCounts[i] = 0;
        TotalAbsorbed = 0;

        for (int i = 0; i < count; i++)
        {
            if (p[i].alive < 0.5f)
            {
                int t = p[i].typeIndex;
                if (t >= 0 && t < AbsorbedCounts.Length) AbsorbedCounts[t]++;
                TotalAbsorbed++;
            }
        }
    }

    public void SetTargetType(int typeIndex) { targetTypeIndex = typeIndex; }
}
