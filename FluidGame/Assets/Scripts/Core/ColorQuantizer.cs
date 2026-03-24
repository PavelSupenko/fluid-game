using UnityEngine;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Median Cut color quantization algorithm.
/// Takes an array of pixel colors and reduces them to a target palette size.
/// Returns the palette and a mapping from each input pixel to its closest palette index.
///
/// The algorithm works by repeatedly splitting the color "box" with the widest
/// range along its widest color channel, until we have the desired number of boxes.
/// Each box's average color becomes one palette entry.
/// </summary>
public static class ColorQuantizer
{
    /// <summary>
    /// Result of quantization: the reduced palette and per-pixel assignments.
    /// </summary>
    public struct QuantizeResult
    {
        public Color[] palette;     // The N quantized colors
        public int[] assignments;   // Per-pixel index into palette (same length as input)
    }

    /// <summary>
    /// Quantize an array of colors down to targetColors distinct colors.
    /// </summary>
    /// <param name="pixels">Input pixel colors (e.g. from Texture2D.GetPixels)</param>
    /// <param name="targetColors">Desired palette size (e.g. 8 or 16)</param>
    /// <param name="pixels">Input pixel colors</param>
    /// <param name="targetColors">Desired palette size (e.g. 8 or 16)</param>
    /// <param name="minPercentage">Colors occupying less than this % of pixels get merged
    /// into their nearest surviving color. 0 = keep all.</param>
    public static QuantizeResult Quantize(Color[] pixels, int targetColors, float minPercentage = 0f)
    {
        targetColors = Mathf.Clamp(targetColors, 2, 32);

        // Build initial list of pixel indices (skip fully transparent pixels)
        var indices = new List<int>(pixels.Length);
        for (int i = 0; i < pixels.Length; i++)
        {
            if (pixels[i].a > 0.1f)
                indices.Add(i);
        }

        int opaqueCount = indices.Count;

        // Start with one box containing all pixels
        var boxes = new List<ColorBox>();
        boxes.Add(new ColorBox(pixels, indices));

        // Repeatedly split the box with the largest range until we have enough
        while (boxes.Count < targetColors)
        {
            int bestIdx = 0;
            float bestRange = 0f;
            for (int i = 0; i < boxes.Count; i++)
            {
                float range = boxes[i].GetWidestRange();
                if (range > bestRange && boxes[i].pixelIndices.Count > 1)
                {
                    bestRange = range;
                    bestIdx = i;
                }
            }

            if (bestRange < 0.001f) break;

            var toSplit = boxes[bestIdx];
            boxes.RemoveAt(bestIdx);

            var (boxA, boxB) = toSplit.Split();
            if (boxA.pixelIndices.Count > 0) boxes.Add(boxA);
            if (boxB.pixelIndices.Count > 0) boxes.Add(boxB);
        }

        // Build initial palette from box averages
        Color[] rawPalette = new Color[boxes.Count];
        for (int i = 0; i < boxes.Count; i++)
            rawPalette[i] = boxes[i].GetAverageColor();

        // Assign each pixel to its nearest color in the raw palette
        int[] rawAssignments = new int[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
            rawAssignments[i] = FindNearestColor(pixels[i], rawPalette);

        // ── Filter rare colors ──
        // Count how many pixels belong to each palette color
        if (minPercentage > 0f && opaqueCount > 0)
        {
            int[] colorCounts = new int[rawPalette.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                if (pixels[i].a > 0.1f)
                    colorCounts[rawAssignments[i]]++;
            }

            float minCount = opaqueCount * (minPercentage / 100f);

            // Mark which colors survive
            bool[] survives = new bool[rawPalette.Length];
            int survivorCount = 0;
            for (int i = 0; i < rawPalette.Length; i++)
            {
                survives[i] = colorCounts[i] >= minCount;
                if (survives[i]) survivorCount++;
            }

            // Ensure at least 2 colors survive
            if (survivorCount < 2)
            {
                // Keep the two largest
                var sorted = new List<int>();
                for (int i = 0; i < rawPalette.Length; i++) sorted.Add(i);
                sorted.Sort((a, b) => colorCounts[b].CompareTo(colorCounts[a]));
                for (int i = 0; i < rawPalette.Length; i++) survives[i] = false;
                survives[sorted[0]] = true;
                if (sorted.Count > 1) survives[sorted[1]] = true;
                survivorCount = Mathf.Min(2, sorted.Count);
            }

            if (survivorCount < rawPalette.Length)
            {
                // Build compact palette from survivors
                Color[] filteredPalette = new Color[survivorCount];
                int[] oldToNew = new int[rawPalette.Length];
                int newIdx = 0;
                for (int i = 0; i < rawPalette.Length; i++)
                {
                    if (survives[i])
                    {
                        filteredPalette[newIdx] = rawPalette[i];
                        oldToNew[i] = newIdx;
                        newIdx++;
                    }
                    else
                    {
                        oldToNew[i] = -1; // Will be reassigned
                    }
                }

                // Map dead colors → nearest surviving color
                for (int i = 0; i < rawPalette.Length; i++)
                {
                    if (oldToNew[i] < 0)
                        oldToNew[i] = FindNearestColor(rawPalette[i], filteredPalette);
                }

                // Remap all assignments
                for (int i = 0; i < pixels.Length; i++)
                    rawAssignments[i] = oldToNew[rawAssignments[i]];

                int removed = rawPalette.Length - survivorCount;
                Debug.Log($"[ColorQuantizer] Filtered {removed} rare colors " +
                          $"(< {minPercentage:F1}% = {minCount:F0} pixels). " +
                          $"{survivorCount} colors remain.");

                rawPalette = filteredPalette;
            }
        }

        Debug.Log($"[ColorQuantizer] Quantized to {rawPalette.Length} colors " +
                  $"from {pixels.Length} pixels ({opaqueCount} non-transparent)");

        return new QuantizeResult
        {
            palette = rawPalette,
            assignments = rawAssignments
        };
    }

    /// <summary>
    /// Finds the index of the closest color in the palette using squared distance in RGB.
    /// </summary>
    static int FindNearestColor(Color pixel, Color[] palette)
    {
        float bestDist = float.MaxValue;
        int bestIdx = 0;

        for (int i = 0; i < palette.Length; i++)
        {
            float dr = pixel.r - palette[i].r;
            float dg = pixel.g - palette[i].g;
            float db = pixel.b - palette[i].b;
            float dist = dr * dr + dg * dg + db * db;

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /// <summary>
    /// Represents an axis-aligned bounding box in RGB color space
    /// containing a set of pixel indices.
    /// </summary>
    private class ColorBox
    {
        public List<int> pixelIndices;
        private Color[] allPixels;

        // Cached min/max per channel
        private float rMin, rMax, gMin, gMax, bMin, bMax;
        private int widestChannel; // 0=R, 1=G, 2=B

        public ColorBox(Color[] pixels, List<int> indices)
        {
            allPixels = pixels;
            pixelIndices = indices;
            ComputeBounds();
        }

        void ComputeBounds()
        {
            rMin = gMin = bMin = float.MaxValue;
            rMax = gMax = bMax = float.MinValue;

            for (int i = 0; i < pixelIndices.Count; i++)
            {
                Color c = allPixels[pixelIndices[i]];
                if (c.r < rMin) rMin = c.r; if (c.r > rMax) rMax = c.r;
                if (c.g < gMin) gMin = c.g; if (c.g > gMax) gMax = c.g;
                if (c.b < bMin) bMin = c.b; if (c.b > bMax) bMax = c.b;
            }

            float rRange = rMax - rMin;
            float gRange = gMax - gMin;
            float bRange = bMax - bMin;

            widestChannel = (rRange >= gRange && rRange >= bRange) ? 0
                          : (gRange >= bRange) ? 1 : 2;
        }

        public float GetWidestRange()
        {
            return widestChannel switch
            {
                0 => rMax - rMin,
                1 => gMax - gMin,
                _ => bMax - bMin,
            };
        }

        /// <summary>
        /// Splits this box into two halves along the widest channel at the median.
        /// </summary>
        public (ColorBox, ColorBox) Split()
        {
            // Sort pixel indices by the widest channel
            int ch = widestChannel;
            pixelIndices.Sort((a, b) =>
            {
                float va = GetChannel(allPixels[a], ch);
                float vb = GetChannel(allPixels[b], ch);
                return va.CompareTo(vb);
            });

            int mid = pixelIndices.Count / 2;
            var listA = pixelIndices.GetRange(0, mid);
            var listB = pixelIndices.GetRange(mid, pixelIndices.Count - mid);

            return (new ColorBox(allPixels, listA), new ColorBox(allPixels, listB));
        }

        public Color GetAverageColor()
        {
            float r = 0, g = 0, b = 0;
            for (int i = 0; i < pixelIndices.Count; i++)
            {
                Color c = allPixels[pixelIndices[i]];
                r += c.r; g += c.g; b += c.b;
            }
            float n = pixelIndices.Count;
            return new Color(r / n, g / n, b / n, 1f);
        }

        static float GetChannel(Color c, int ch)
        {
            return ch switch { 0 => c.r, 1 => c.g, _ => c.b };
        }
    }
}