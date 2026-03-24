using UnityEngine;

/// <summary>
/// Flask controller for sucking up particles. Works with FluidSimulationJobs.
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

    public int[] AbsorbedCounts { get; private set; }
    public int TotalAbsorbed { get; private set; }
    public bool IsSucking { get; private set; }
    public Vector2 FlaskWorldPos { get; private set; }

    private FluidSimulationJobs sim;
    private Camera mainCam;

    void Start()
    {
        mainCam = Camera.main;
        sim = FindObjectOfType<FluidSimulationJobs>();

        if (sim == null || !sim.enabled)
        {
            Debug.LogError("[FlaskController] No active FluidSimulationJobs found!");
            enabled = false;
            return;
        }

        AbsorbedCounts = new int[Mathf.Max(sim.fluidTypes.Length, 1)];
    }

    void Update()
    {
        UpdateFlaskPosition();
        IsSucking = Input.GetMouseButton(0);
        PassToSim();

        if (Time.frameCount % 15 == 0)
            CountAbsorbed();
    }

    void UpdateFlaskPosition()
    {
        Vector3 mouseScreen = Input.mousePosition;
        mouseScreen.z = Mathf.Abs(mainCam.transform.position.z);
        Vector3 worldPos = mainCam.ScreenToWorldPoint(mouseScreen);
        FlaskWorldPos = new Vector2(worldPos.x, worldPos.y);
    }

    void PassToSim()
    {
        sim.flaskActive = IsSucking;
        sim.flaskPos = new Unity.Mathematics.float2(FlaskWorldPos.x, FlaskWorldPos.y);
        sim.flaskTargetType = targetTypeIndex;
        sim.flaskRadius = suctionRadius;
        sim.flaskAbsorbRadius = absorbRadius;
        sim.flaskStrength = suctionStrength;
    }

    void CountAbsorbed()
    {
        if (sim.Particles == null) return;

        for (int i = 0; i < AbsorbedCounts.Length; i++) AbsorbedCounts[i] = 0;
        TotalAbsorbed = 0;

        for (int i = 0; i < sim.ParticleCount; i++)
        {
            if (sim.Particles[i].alive < 0.5f)
            {
                int t = sim.Particles[i].typeIndex;
                if (t >= 0 && t < AbsorbedCounts.Length) AbsorbedCounts[t]++;
                TotalAbsorbed++;
            }
        }
    }

    public void SetTargetType(int typeIndex) { targetTypeIndex = typeIndex; }
}