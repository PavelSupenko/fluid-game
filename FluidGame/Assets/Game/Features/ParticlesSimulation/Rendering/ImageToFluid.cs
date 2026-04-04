using UnityEngine;
using System;

public struct FluidParticle
{
    public Vector2 position;
    public int typeIndex;
}

[Serializable]
public struct FluidTypeDefinition
{
    public Color color;
}

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
    [Range(2, 200)]
    public int resolution = 80;

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

    public void TryParseImage(Rect? bounds)
    {
        if (sourceImage == null)
        {
            Debug.Log("[ImageToFluid] No source image assigned — simulation will use default grid spawn.");
            return;
        }

        ProcessImage(bounds);
    }

    // ─── Image Processing ────────────────────────────────────────

    void ProcessImage(Rect? bounds)
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
                color = result.palette[i],
            };
        }

        // ── Step 5: Determine container-relative positioning ──
        // Read container bounds from whichever simulation is present
        Vector2 containerMin, containerMax;

        if (bounds != null)
        {
            containerMin = bounds.Value.min;
            containerMax = bounds.Value.max;
        }
        else
        {
            containerMin = new Vector2(-4f, -3f);
            containerMax = new Vector2(4f, 4f);
        }

        float containerW = containerMax.x - containerMin.x;
        float containerH = containerMax.y - containerMin.y;

        // Fit image edge-to-edge: flush to left, right, and bottom of container.
        // Spacing is determined by whichever axis is tighter (width or height).
        // Typically width is the limiting factor for landscape images.
        float spacingX = containerW / sampleW;
        float spacingY = containerH / sampleH;
        ComputedSpacing = Mathf.Min(spacingX, spacingY);

        // Compute actual image dimensions with chosen spacing
        float totalW = sampleW * ComputedSpacing;
        float totalH = sampleH * ComputedSpacing;

        // Center horizontally within container, flush to bottom
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
                    typeIndex = typeIdx,
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