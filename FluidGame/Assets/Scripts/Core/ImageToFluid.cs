using UnityEngine;

/// <summary>
/// Converts a Texture2D into a fluid particle field.
/// 
/// Assign a source image in the inspector. On Awake, this component:
///   1. Reads pixel data from the texture
///   2. Quantizes the palette to targetColorCount colors (median cut)
///   3. Creates FluidTypeDefinition[] with uniform physics but distinct colors
///   4. Creates a FluidParticle[] grid that reproduces the image
///
/// FluidSimulationGPU checks for this component and uses its data instead
/// of the default grid spawn.
///
/// SETUP: Add this component to the same GameObject as FluidSimulationGPU.
///        Assign a Texture2D (must have Read/Write enabled in import settings).
/// </summary>
public class ImageToFluid : MonoBehaviour
{
    [Header("Source Image")]
    [Tooltip("The image to convert into fluid. Must have Read/Write enabled in import settings.")]
    public Texture2D sourceImage;

    [Header("Quantization")]
    [Tooltip("Number of distinct colors in the fluid palette")]
    [Range(2, 16)]
    public int targetColorCount = 8;

    [Tooltip("Colors occupying less than this % of pixels are merged into their nearest major color. " +
             "Eliminates compression artifacts and tiny color slivers.")]
    [Range(0f, 20f)]
    public float minColorPercentage = 5f;

    [Header("Particle Resolution")]
    [Tooltip("Max particles along the wider image axis. Total particles = this² (roughly).")]
    [Range(20, 200)]
    public int resolution = 80;

    [Header("Uniform Physics")]
    [Tooltip("All fluid types share these values so the image stays stable")]
    public float uniformDensity = 2f;
    public float uniformViscosity = 6f;
    public float uniformCohesion = 1f;

    // ─── Output Data (read by FluidSimulationGPU) ────────────────

    /// <summary>True after Awake if image was successfully processed.</summary>
    public bool IsReady { get; private set; }

    /// <summary>Generated particles positioned to reproduce the image.</summary>
    public FluidParticle[] GeneratedParticles { get; private set; }

    /// <summary>Particle count (may be less than resolution² due to transparent pixels).</summary>
    public int GeneratedParticleCount { get; private set; }

    /// <summary>Fluid type definitions with uniform physics and quantized colors.</summary>
    public FluidTypeDefinition[] GeneratedFluidTypes { get; private set; }

    /// <summary>Computed particle spacing based on resolution and container size.</summary>
    public float ComputedSpacing { get; private set; }

    // ─── Lifecycle ───────────────────────────────────────────────

    public void TryParseImage()
    {
        if (sourceImage == null)
        {
            Debug.Log("[ImageToFluid] No source image assigned — simulation will use default grid spawn.");
            return;
        }

        ProcessImage();
    }

    // ─── Image Processing ────────────────────────────────────────

    void ProcessImage()
    {
        // Read pixels
        Color[] pixels;
        int imgWidth, imgHeight;

        try
        {
            pixels = sourceImage.GetPixels();
            imgWidth = sourceImage.width;
            imgHeight = sourceImage.height;

            // GetPixels() on sRGB textures returns linear-space colors.
            // Convert back to gamma space so the fluid matches the original image brightness.
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = pixels[i].gamma;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[ImageToFluid] Failed to read texture. " +
                           $"Make sure Read/Write is enabled in import settings. Error: {e.Message}");
            return;
        }

        Debug.Log($"[ImageToFluid] Source image: {imgWidth}x{imgHeight} ({pixels.Length} pixels)");

        // ── Step 1: Determine sampling resolution ──
        // Scale the image down to fit within 'resolution' particles on the longer axis
        float aspect = (float)imgWidth / imgHeight;
        int sampleW, sampleH;

        if (aspect >= 1f)
        {
            sampleW = resolution;
            sampleH = Mathf.Max(1, Mathf.RoundToInt(resolution / aspect));
        }
        else
        {
            sampleH = resolution;
            sampleW = Mathf.Max(1, Mathf.RoundToInt(resolution * aspect));
        }

        // ── Step 2: Sample the image at particle resolution ──
        Color[] sampledPixels = new Color[sampleW * sampleH];
        for (int y = 0; y < sampleH; y++)
        {
            for (int x = 0; x < sampleW; x++)
            {
                // Map sample coordinates to image coordinates using bilinear-like sampling
                float u = (x + 0.5f) / sampleW;
                float v = (y + 0.5f) / sampleH;
                int ix = Mathf.Clamp(Mathf.FloorToInt(u * imgWidth), 0, imgWidth - 1);
                int iy = Mathf.Clamp(Mathf.FloorToInt(v * imgHeight), 0, imgHeight - 1);

                sampledPixels[y * sampleW + x] = pixels[iy * imgWidth + ix];
            }
        }

        Debug.Log($"[ImageToFluid] Sampled to {sampleW}x{sampleH} = {sampledPixels.Length} samples");

        // ── Step 3: Quantize colors ──
        var result = ColorQuantizer.Quantize(sampledPixels, targetColorCount, minColorPercentage);

        // ── Step 4: Create fluid type definitions (uniform physics, different colors) ──
        GeneratedFluidTypes = new FluidTypeDefinition[result.palette.Length];
        for (int i = 0; i < result.palette.Length; i++)
        {
            GeneratedFluidTypes[i] = new FluidTypeDefinition
            {
                name = $"Color_{i}",
                color = result.palette[i],
                density = uniformDensity,
                viscosity = uniformViscosity,
                cohesion = uniformCohesion
            };
        }

        // ── Step 5: Determine container-relative positioning ──
        // Read container bounds from whichever simulation is present
        Vector2 containerMin, containerMax;

        var gpuSim = GetComponent<FluidSimulationGPU>();
        var jobsSim = GetComponent<FluidSimulationJobs>();

        if (gpuSim != null)
        {
            containerMin = gpuSim.containerMin;
            containerMax = gpuSim.containerMax;
        }
        else if (jobsSim != null)
        {
            containerMin = jobsSim.containerMin;
            containerMax = jobsSim.containerMax;
        }
        else
        {
            containerMin = new Vector2(-4f, -3f);
            containerMax = new Vector2(4f, 4f);
        }

        float containerW = containerMax.x - containerMin.x;
        float containerH = containerMax.y - containerMin.y;

        // Fit the image in the container with small margin on sides
        float margin = 0.2f;
        float availW = containerW - margin * 2f;
        float availH = containerH - margin;  // No top margin needed, bottom flush

        // Compute spacing so the image fits within available space
        float spacingX = availW / sampleW;
        float spacingY = availH / sampleH;
        ComputedSpacing = Mathf.Min(spacingX, spacingY);

        // Center horizontally, align BOTTOM of image to container bottom
        float totalW = sampleW * ComputedSpacing;
        float totalH = sampleH * ComputedSpacing;
        float originX = containerMin.x + (containerW - totalW) * 0.5f + ComputedSpacing * 0.5f;
        float originY = containerMin.y + ComputedSpacing * 0.5f; // Bottom-aligned

        // ── Step 6: Create particles ──
        // First pass: count non-transparent pixels
        int count = 0;
        for (int i = 0; i < sampledPixels.Length; i++)
        {
            if (sampledPixels[i].a > 0.1f) count++;
        }

        GeneratedParticles = new FluidParticle[count];
        int idx = 0;

        for (int y = 0; y < sampleH; y++)
        {
            for (int x = 0; x < sampleW; x++)
            {
                int si = y * sampleW + x;
                if (sampledPixels[si].a <= 0.1f) continue;

                int typeIdx = result.assignments[si];

                GeneratedParticles[idx] = new FluidParticle
                {
                    position = new Vector2(
                        originX + x * ComputedSpacing,
                        originY + y * ComputedSpacing
                    ),
                    velocity = Vector2.zero,
                    typeIndex = typeIdx,
                    density = 0f,
                    pressure = 0f,
                    alive = 1f,
                    color = result.palette[typeIdx]
                };

                idx++;
            }
        }

        GeneratedParticleCount = count;
        IsReady = true;

        // Log palette summary
        string paletteSummary = "[ImageToFluid] Palette: ";
        int[] typeCounts = new int[result.palette.Length];
        for (int i = 0; i < count; i++)
            typeCounts[GeneratedParticles[i].typeIndex]++;
        for (int i = 0; i < result.palette.Length; i++)
            paletteSummary += $"#{ColorUtility.ToHtmlStringRGB(result.palette[i])}({typeCounts[i]}) ";

        Debug.Log(paletteSummary);
        Debug.Log($"[ImageToFluid] Generated {count} particles, spacing={ComputedSpacing:F4}, " +
                  $"grid={sampleW}x{sampleH}, palette={result.palette.Length} colors");
    }
}