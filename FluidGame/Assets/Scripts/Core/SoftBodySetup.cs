using UnityEngine;
using System.Collections.Generic;
using Unity.Mathematics;

/// <summary>
/// Sets up soft body data for particles. Two modes:
///
/// 1. MASK MODE (autoSegmentByColor = false):
///    Uses a mask texture where each color = one soft body.
///
/// 2. AUTO-SEGMENT MODE (autoSegmentByColor = true):
///    Automatically detects connected regions of the same color in the quantized image.
///    Each contiguous region of the same typeIndex becomes a separate soft body.
///    E.g., a red circle on a green background = 2 bodies.
///
/// If neither mask is assigned nor auto-segment is enabled, all particles = one body.
/// </summary>
public class SoftBodySetup : MonoBehaviour
{
    [Header("Segmentation Mode")]
    [Tooltip("When ON: automatically split into bodies by connected color regions. " +
             "When OFF: use mask texture to define bodies.")]
    public bool autoSegmentByColor = true;

    [Tooltip("Minimum particles for a region to become its own body (auto-segment mode). " +
             "Smaller regions are merged into the nearest neighbor body.")]
    public int minRegionSize = 10;

    [Header("Mask (used when autoSegmentByColor is OFF)")]
    [Tooltip("Mask texture where each color defines a separate soft body.")]
    public Texture2D maskTexture;

    [Tooltip("Color distance threshold for grouping mask colors")]
    [Range(0.01f, 0.3f)]
    public float maskColorThreshold = 0.1f;

    [Header("Spring Properties")]
    [Range(1, 2)]
    public int connectionRings = 1;

    [Range(1.5f, 5f)]
    public float defaultBreakThreshold = 2.5f;

    // ─── Output Data ─────────────────────────────────────────────

    public bool IsReady { get; private set; }
    public int[] BodyIndices { get; private set; }
    public SoftBodySpring[] Springs { get; private set; }
    public int SpringCount { get; private set; }
    public int BodyCount { get; private set; }
    public int[] ParticlesPerBody { get; private set; }
    public Color[] BodyColors { get; private set; }

    // ─── Public API ──────────────────────────────────────────────

    public void Process(ImageToFluid imageSource)
    {
        if (imageSource == null || !imageSource.IsReady)
        {
            CreateSingleBody(imageSource);
            return;
        }

        if (autoSegmentByColor)
        {
            ProcessAutoSegment(imageSource);
        }
        else if (maskTexture != null)
        {
            ProcessMask(imageSource);
        }
        else
        {
            Debug.Log("[SoftBodySetup] No mask and auto-segment off — single body.");
            CreateSingleBody(imageSource);
        }
    }

    // ═════════════════════════════════════════════════════════════
    //  AUTO-SEGMENT MODE — flood fill by typeIndex
    // ═════════════════════════════════════════════════════════════

    /// <summary>
    /// Uses flood fill on the quantized pixel grid to find connected regions
    /// of the same typeIndex. Each region = one soft body.
    /// </summary>
    void ProcessAutoSegment(ImageToFluid source)
    {
        int particleCount = source.GeneratedParticleCount;
        var particles = source.GeneratedParticles;
        float spacing = source.ComputedSpacing;
        int w = source.SampleWidth;
        int h = source.SampleHeight;
        int[] typeGrid = source.PixelTypeGrid;        // Per-pixel typeIndex, -1 = transparent
        int[] pixToParticle = source.PixelToParticle;  // Per-pixel particle index, -1 = none

        int totalPixels = w * h;

        // ── Step 1: Flood fill to find connected components ──
        int[] pixelBodyMap = new int[totalPixels]; // body index per pixel
        for (int i = 0; i < totalPixels; i++) pixelBodyMap[i] = -1;

        var bodyColorList = new List<Color>();
        var bodySizeList = new List<int>();
        var bodyPixelLists = new List<List<int>>(); // pixel indices per body

        // Queue for BFS flood fill
        var queue = new Queue<int>(256);

        for (int startPixel = 0; startPixel < totalPixels; startPixel++)
        {
            if (pixelBodyMap[startPixel] >= 0) continue; // Already visited
            if (typeGrid[startPixel] < 0) continue;       // Transparent

            int targetType = typeGrid[startPixel];
            int bodyIdx = bodyColorList.Count;

            var pixelsInBody = new List<int>();

            // BFS flood fill (4-connected: up, down, left, right)
            queue.Clear();
            queue.Enqueue(startPixel);
            pixelBodyMap[startPixel] = bodyIdx;

            while (queue.Count > 0)
            {
                int px = queue.Dequeue();
                pixelsInBody.Add(px);

                int px_x = px % w;
                int px_y = px / w;

                // Check 4 neighbors
                TryEnqueue(px_x - 1, px_y, w, h, targetType, bodyIdx, typeGrid, pixelBodyMap, queue);
                TryEnqueue(px_x + 1, px_y, w, h, targetType, bodyIdx, typeGrid, pixelBodyMap, queue);
                TryEnqueue(px_x, px_y - 1, w, h, targetType, bodyIdx, typeGrid, pixelBodyMap, queue);
                TryEnqueue(px_x, px_y + 1, w, h, targetType, bodyIdx, typeGrid, pixelBodyMap, queue);
            }

            // Determine body color from the first particle in this region
            Color bodyColor = Color.white;
            for (int i = 0; i < pixelsInBody.Count; i++)
            {
                int pi = pixToParticle[pixelsInBody[i]];
                if (pi >= 0)
                {
                    bodyColor = particles[pi].color;
                    break;
                }
            }

            bodyColorList.Add(bodyColor);
            bodySizeList.Add(pixelsInBody.Count);
            bodyPixelLists.Add(pixelsInBody);
        }

        Debug.Log($"[SoftBodySetup] Auto-segment found {bodyColorList.Count} raw regions");

        // ── Step 2: Merge tiny regions into nearest larger neighbor ──
        int rawCount = bodyColorList.Count;
        int[] bodyRemap = new int[rawCount]; // maps old body index → final body index
        for (int i = 0; i < rawCount; i++) bodyRemap[i] = i;

        // Mark small regions for merging
        for (int b = 0; b < rawCount; b++)
        {
            if (bodySizeList[b] >= minRegionSize) continue;

            // Find the nearest large-enough body by checking neighbor pixels
            int bestNeighborBody = -1;
            var px_list = bodyPixelLists[b];

            for (int pi = 0; pi < px_list.Count && bestNeighborBody < 0; pi++)
            {
                int px = px_list[pi];
                int px_x = px % w;
                int px_y = px / w;

                // Check 4 neighbors for a different body that is large enough
                bestNeighborBody = FindLargeNeighborBody(px_x - 1, px_y, w, h, b, pixelBodyMap, bodySizeList);
                if (bestNeighborBody >= 0) break;
                bestNeighborBody = FindLargeNeighborBody(px_x + 1, px_y, w, h, b, pixelBodyMap, bodySizeList);
                if (bestNeighborBody >= 0) break;
                bestNeighborBody = FindLargeNeighborBody(px_x, px_y - 1, w, h, b, pixelBodyMap, bodySizeList);
                if (bestNeighborBody >= 0) break;
                bestNeighborBody = FindLargeNeighborBody(px_x, px_y + 1, w, h, b, pixelBodyMap, bodySizeList);
            }

            if (bestNeighborBody >= 0)
                bodyRemap[b] = bestNeighborBody;
        }

        // Compact body indices: remap to contiguous 0..N
        var finalBodyIds = new Dictionary<int, int>();
        int nextFinalId = 0;

        for (int b = 0; b < rawCount; b++)
        {
            int target = bodyRemap[b];
            // Follow remap chain
            while (bodyRemap[target] != target) target = bodyRemap[target];
            bodyRemap[b] = target;

            if (!finalBodyIds.ContainsKey(target))
                finalBodyIds[target] = nextFinalId++;
        }

        BodyCount = nextFinalId;
        BodyColors = new Color[BodyCount];
        ParticlesPerBody = new int[BodyCount];

        for (int b = 0; b < rawCount; b++)
        {
            int finalId = finalBodyIds[bodyRemap[b]];
            if (bodySizeList[b] >= minRegionSize || bodyRemap[b] == b)
                BodyColors[finalId] = bodyColorList[b];
        }

        // ── Step 3: Map pixel body indices → particle body indices ──
        BodyIndices = new int[particleCount];

        for (int px = 0; px < totalPixels; px++)
        {
            int pi = pixToParticle[px];
            if (pi < 0) continue;

            int rawBody = pixelBodyMap[px];
            if (rawBody < 0)
            {
                BodyIndices[pi] = 0;
                continue;
            }

            int finalId = finalBodyIds[bodyRemap[rawBody]];
            BodyIndices[pi] = finalId;
            ParticlesPerBody[finalId]++;
        }

        // ── Step 4: Create springs ──
        CreateSprings(particles, spacing);

        // Log results
        Debug.Log($"[SoftBodySetup] Auto-segment: {BodyCount} bodies after merging " +
                  $"(min region size = {minRegionSize})");
        for (int b = 0; b < BodyCount; b++)
            Debug.Log($"  Body {b}: {ParticlesPerBody[b]} particles, " +
                      $"color=#{ColorUtility.ToHtmlStringRGB(BodyColors[b])}");

        IsReady = true;
    }

    void TryEnqueue(int x, int y, int w, int h, int targetType,
                    int bodyIdx, int[] typeGrid, int[] pixelBodyMap, Queue<int> queue)
    {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        int idx = y * w + x;
        if (pixelBodyMap[idx] >= 0) return;     // Already visited
        if (typeGrid[idx] != targetType) return; // Different type

        pixelBodyMap[idx] = bodyIdx;
        queue.Enqueue(idx);
    }

    int FindLargeNeighborBody(int x, int y, int w, int h,
                               int currentBody, int[] pixelBodyMap, List<int> bodySizes)
    {
        if (x < 0 || x >= w || y < 0 || y >= h) return -1;
        int idx = y * w + x;
        int nb = pixelBodyMap[idx];
        if (nb < 0 || nb == currentBody) return -1;
        if (bodySizes[nb] < minRegionSize) return -1;
        return nb;
    }

    // ═════════════════════════════════════════════════════════════
    //  MASK MODE — uses mask texture colors
    // ═════════════════════════════════════════════════════════════

    void ProcessMask(ImageToFluid imageSource)
    {
        int particleCount = imageSource.GeneratedParticleCount;
        var particles = imageSource.GeneratedParticles;
        float spacing = imageSource.ComputedSpacing;
        int w = imageSource.SampleWidth;
        int h = imageSource.SampleHeight;
        int[] pixToParticle = imageSource.PixelToParticle;

        // Sample mask at same resolution
        Color[] maskPixels;
        try { maskPixels = maskTexture.GetPixels(); }
        catch (System.Exception e)
        {
            Debug.LogError($"[SoftBodySetup] Mask read failed: {e.Message}");
            CreateSingleBody(imageSource);
            return;
        }

        int imgW = maskTexture.width, imgH = maskTexture.height;
        Color[] sampledMask = new Color[w * h];

        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            float u = (x + 0.5f) / w;
            float v = (y + 0.5f) / h;
            int ix = Mathf.Clamp(Mathf.FloorToInt(u * imgW), 0, imgW - 1);
            int iy = Mathf.Clamp(Mathf.FloorToInt(v * imgH), 0, imgH - 1);
            sampledMask[y * w + x] = maskPixels[iy * imgW + ix];
        }

        // Identify body colors
        var bodyColorList = new List<Color>();
        int[] maskBodyMap = new int[w * h];

        for (int i = 0; i < sampledMask.Length; i++)
        {
            Color c = sampledMask[i];
            if (c.a < 0.1f || (c.r < 0.05f && c.g < 0.05f && c.b < 0.05f))
            { maskBodyMap[i] = -1; continue; }

            int bodyIdx = -1;
            for (int b = 0; b < bodyColorList.Count; b++)
            {
                float dr = c.r - bodyColorList[b].r;
                float dg = c.g - bodyColorList[b].g;
                float db = c.b - bodyColorList[b].b;
                if (dr * dr + dg * dg + db * db < maskColorThreshold * maskColorThreshold)
                { bodyIdx = b; break; }
            }

            if (bodyIdx < 0)
            {
                bodyIdx = bodyColorList.Count;
                bodyColorList.Add(c);
            }
            maskBodyMap[i] = bodyIdx;
        }

        BodyCount = bodyColorList.Count;
        BodyColors = bodyColorList.ToArray();
        BodyIndices = new int[particleCount];
        ParticlesPerBody = new int[BodyCount];

        // Map pixels → particles
        for (int px = 0; px < w * h; px++)
        {
            int pi = pixToParticle[px];
            if (pi < 0) continue;

            int bodyIdx = maskBodyMap[px];
            if (bodyIdx < 0) bodyIdx = 0;

            BodyIndices[pi] = bodyIdx;
            if (bodyIdx < BodyCount) ParticlesPerBody[bodyIdx]++;
        }

        CreateSprings(particles, spacing);

        Debug.Log($"[SoftBodySetup] Mask mode: {BodyCount} bodies");
        for (int b = 0; b < BodyCount; b++)
            Debug.Log($"  Body {b}: {ParticlesPerBody[b]} particles");

        IsReady = true;
    }

    // ═════════════════════════════════════════════════════════════
    //  SPRING GENERATION (shared by all modes)
    // ═════════════════════════════════════════════════════════════

    void CreateSprings(FluidParticle[] particles, float spacing)
    {
        int particleCount = particles.Length;
        float maxDist = spacing * (connectionRings + 0.5f);
        float maxDistSqr = maxDist * maxDist;

        var springList = new List<SoftBodySpring>();

        float cellSize = maxDist;
        float minX = float.MaxValue, minY = float.MaxValue;
        for (int i = 0; i < particleCount; i++)
        {
            if (particles[i].position.x < minX) minX = particles[i].position.x;
            if (particles[i].position.y < minY) minY = particles[i].position.y;
        }

        var gridDict = new Dictionary<int, List<int>>();
        int gridW = 1000;

        for (int i = 0; i < particleCount; i++)
        {
            int cx = Mathf.FloorToInt((particles[i].position.x - minX) / cellSize);
            int cy = Mathf.FloorToInt((particles[i].position.y - minY) / cellSize);
            int key = cy * gridW + cx;
            if (!gridDict.ContainsKey(key)) gridDict[key] = new List<int>();
            gridDict[key].Add(i);
        }

        for (int i = 0; i < particleCount; i++)
        {
            if (BodyIndices[i] < 0) continue;

            int cx = Mathf.FloorToInt((particles[i].position.x - minX) / cellSize);
            int cy = Mathf.FloorToInt((particles[i].position.y - minY) / cellSize);

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                int key = (cy + dy) * gridW + (cx + dx);
                if (!gridDict.TryGetValue(key, out var cell)) continue;

                for (int n = 0; n < cell.Count; n++)
                {
                    int j = cell[n];
                    if (j <= i) continue;
                    if (BodyIndices[j] != BodyIndices[i]) continue;

                    Vector2 diff = particles[i].position - particles[j].position;
                    float distSqr = diff.sqrMagnitude;

                    if (distSqr < maxDistSqr && distSqr > 0.0001f)
                    {
                        springList.Add(new SoftBodySpring
                        {
                            particleA = i, particleB = j,
                            restLength = Mathf.Sqrt(distSqr),
                            breakThreshold = defaultBreakThreshold,
                            alive = 1
                        });
                    }
                }
            }
        }

        Springs = springList.ToArray();
        SpringCount = Springs.Length;

        float avg = particleCount > 0 ? (float)SpringCount * 2f / particleCount : 0f;
        Debug.Log($"[SoftBodySetup] {SpringCount} springs (avg {avg:F1}/particle)");
    }

    // ═════════════════════════════════════════════════════════════
    //  SINGLE BODY FALLBACK
    // ═════════════════════════════════════════════════════════════

    void CreateSingleBody(ImageToFluid imageSource)
    {
        if (imageSource == null || !imageSource.IsReady)
        {
            BodyIndices = new int[0];
            Springs = new SoftBodySpring[0];
            SpringCount = 0; BodyCount = 0;
            IsReady = true;
            return;
        }

        int particleCount = imageSource.GeneratedParticleCount;
        BodyCount = 1;
        BodyColors = new Color[] { Color.white };
        BodyIndices = new int[particleCount];
        ParticlesPerBody = new int[] { particleCount };
        for (int i = 0; i < particleCount; i++) BodyIndices[i] = 0;

        CreateSprings(imageSource.GeneratedParticles, imageSource.ComputedSpacing);
        IsReady = true;
    }
}