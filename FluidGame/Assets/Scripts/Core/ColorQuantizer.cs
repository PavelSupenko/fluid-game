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
    public static QuantizeResult Quantize(Color[] pixels, int targetColors)
    {
        targetColors = Mathf.Clamp(targetColors, 2, 32);

        // Build initial list of pixel indices (skip fully transparent pixels)
        var indices = new List<int>(pixels.Length);
        for (int i = 0; i < pixels.Length; i++)
        {
            if (pixels[i].a > 0.1f)
                indices.Add(i);
        }

        // Start with one box containing all pixels
        var boxes = new List<ColorBox>();
        boxes.Add(new ColorBox(pixels, indices));

        // Repeatedly split the box with the largest range until we have enough
        while (boxes.Count < targetColors)
        {
            // Find the box with the widest color range
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

            // If no box can be split further, stop early
            if (bestRange < 0.001f) break;

            // Split the widest box along its widest channel
            var toSplit = boxes[bestIdx];
            boxes.RemoveAt(bestIdx);

            var (boxA, boxB) = toSplit.Split();
            if (boxA.pixelIndices.Count > 0) boxes.Add(boxA);
            if (boxB.pixelIndices.Count > 0) boxes.Add(boxB);
        }

        // Build palette from box averages
        Color[] palette = new Color[boxes.Count];
        for (int i = 0; i < boxes.Count; i++)
        {
            palette[i] = boxes[i].GetAverageColor();
        }

        // Assign each pixel to its nearest palette color
        int[] assignments = new int[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            assignments[i] = FindNearestColor(pixels[i], palette);
        }

        Debug.Log($"[ColorQuantizer] Quantized to {palette.Length} colors " +
                  $"from {pixels.Length} pixels ({indices.Count} non-transparent)");

        return new QuantizeResult
        {
            palette = palette,
            assignments = assignments
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
