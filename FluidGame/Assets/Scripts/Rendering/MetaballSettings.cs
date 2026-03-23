using UnityEngine;

/// <summary>
/// Holds metaball rendering settings. Attach this to the Main Camera.
/// The MetaballRenderFeature (on your URP Renderer Asset) reads these values at runtime.
/// </summary>
public class MetaballSettings : MonoBehaviour
{
    [Header("Enable / Disable")]
    [Tooltip("Toggle metaball rendering on/off")]
    public bool showMetaballs = true;

    [Header("Splat Settings")]
    [Tooltip("Size of each particle's gaussian blob. Larger = more merging.")]
    public float splatScale = 0.35f;

    [Tooltip("Controls falloff sharpness. Lower = softer blobs, more merging.")]
    [Range(1f, 8f)]
    public float blobSharpness = 3f;

    [Tooltip("Render target resolution multiplier (1 = full res, 0.5 = half)")]
    [Range(0.25f, 1f)]
    public float resolutionScale = 0.75f;

    [Header("Composite Settings")]
    [Tooltip("How much accumulated weight = solid fluid. Lower = thicker fluid.")]
    [Range(0.01f, 2f)]
    public float threshold = 0.35f;

    [Tooltip("Smoothness of the fluid edge. Higher = softer boundary.")]
    [Range(0.01f, 0.5f)]
    public float edgeSoftness = 0.08f;

    [Tooltip("Bright rim at fluid edges for a glossy look")]
    [Range(0f, 1f)]
    public float edgeHighlight = 0.25f;

    [Tooltip("Color vibrancy boost")]
    [Range(0.5f, 2f)]
    public float colorSaturation = 1.3f;
}
