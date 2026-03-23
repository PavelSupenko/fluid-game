using UnityEngine;

/// <summary>
/// Flask controller for sucking up fluid particles.
///
/// Click and hold the mouse to activate suction at the cursor position.
/// Only particles matching the selected target type are pulled and absorbed.
/// The flask tracks how many particles of each type have been collected.
///
/// SETUP: Add this to the same GameObject as FluidSimulationGPU.
///        FlaskUI (separate component) provides the color selection buttons.
/// </summary>
[RequireComponent(typeof(FluidSimulationGPU))]
public class FlaskController : MonoBehaviour
{
    [Header("Suction Settings")]
    [Tooltip("Outer radius: particles within this distance get pulled toward the flask")]
    public float suctionRadius = 1.0f;

    [Tooltip("Inner radius: particles this close to the cursor get absorbed")]
    public float absorbRadius = 0.15f;

    [Tooltip("Pull force strength. Higher = faster suction.")]
    public float suctionStrength = 80f;

    [Header("Capacity")]
    [Tooltip("When ON, each flask has a maximum number of particles it can hold")]
    public bool limitCapacity = false;

    [Tooltip("Max particles per flask (only used when limitCapacity is ON)")]
    public int maxCapacity = 500;

    [Header("Target")]
    [Tooltip("Which fluid type index to absorb. -1 = absorb all types.")]
    public int targetTypeIndex = 0;

    // ─── Public State ────────────────────────────────────────────

    /// <summary>
    /// Per-type count of absorbed particles. Index = typeIndex.
    /// Updated every readback frame.
    /// </summary>
    public int[] AbsorbedCounts { get; private set; }

    /// <summary>Total absorbed across all types.</summary>
    public int TotalAbsorbed { get; private set; }

    /// <summary>Is suction currently active (mouse held down)?</summary>
    public bool IsSucking { get; private set; }

    /// <summary>Current world position of the flask cursor.</summary>
    public Vector2 FlaskWorldPos { get; private set; }

    // ─── Internals ───────────────────────────────────────────────

    private FluidSimulationGPU sim;
    private Camera mainCam;
    private int lastCountFrame = -1;

    // ─── Lifecycle ───────────────────────────────────────────────

    void Start()
    {
        sim = GetComponent<FluidSimulationGPU>();
        mainCam = Camera.main;

        // Initialize per-type counters
        int typeCount = Mathf.Max(sim.fluidTypes.Length, 1);
        AbsorbedCounts = new int[typeCount];
    }

    void Update()
    {
        UpdateFlaskPosition();
        UpdateSuctionState();
        PassSuctionUniforms();

        // Count absorbed particles periodically (when sim does readback)
        if (Time.frameCount != lastCountFrame && sim.Particles != null)
        {
            // Only recount every ~15 frames to match sim readback interval
            if (Time.frameCount % 15 == 0)
            {
                CountAbsorbed();
                lastCountFrame = Time.frameCount;
            }
        }
    }

    // ─── Input ───────────────────────────────────────────────────

    void UpdateFlaskPosition()
    {
        // Convert mouse screen position to world coordinates
        Vector3 mouseScreen = Input.mousePosition;
        mouseScreen.z = Mathf.Abs(mainCam.transform.position.z);
        Vector3 worldPos = mainCam.ScreenToWorldPoint(mouseScreen);
        FlaskWorldPos = new Vector2(worldPos.x, worldPos.y);
    }

    void UpdateSuctionState()
    {
        // Left mouse button = suction
        IsSucking = Input.GetMouseButton(0);
    }

    // ─── GPU Communication ───────────────────────────────────────

    /// <summary>
    /// Passes flask parameters to the compute shader every frame.
    /// When not sucking, flaskActive = 0 and the GPU skips all suction logic.
    /// </summary>
    void PassSuctionUniforms()
    {
        var cs = sim.computeShader;
        if (cs == null) return;

        cs.SetFloat("flaskActive", IsSucking ? 1f : 0f);
        cs.SetVector("flaskPos", new Vector4(FlaskWorldPos.x, FlaskWorldPos.y, 0, 0));
        cs.SetFloat("flaskTargetType", (float)targetTypeIndex);
        cs.SetFloat("flaskRadius", suctionRadius);
        cs.SetFloat("flaskAbsorbRadius", absorbRadius);
        cs.SetFloat("flaskStrength", suctionStrength);
    }

    // ─── Counting ────────────────────────────────────────────────

    /// <summary>
    /// Counts how many particles have been absorbed per type.
    /// Uses the CPU-side Particles array (updated periodically via readback).
    /// </summary>
    void CountAbsorbed()
    {
        // Reset counters
        for (int i = 0; i < AbsorbedCounts.Length; i++)
            AbsorbedCounts[i] = 0;

        TotalAbsorbed = 0;

        for (int i = 0; i < sim.ParticleCount; i++)
        {
            var p = sim.Particles[i];
            if (p.alive < 0.5f)
            {
                int t = p.typeIndex;
                if (t >= 0 && t < AbsorbedCounts.Length)
                    AbsorbedCounts[t]++;
                TotalAbsorbed++;
            }
        }
    }

    // ─── Public API ──────────────────────────────────────────────

    /// <summary>
    /// Sets the target type for suction. Called by FlaskUI when user clicks a color button.
    /// </summary>
    public void SetTargetType(int typeIndex)
    {
        targetTypeIndex = typeIndex;
    }
}
