using UnityEngine;
using System.Collections.Generic;
using Unity.Mathematics;

/// <summary>
/// Reads a mask texture and sets up soft body data for particles.
///
/// MASK FORMAT:
///   - Each distinct color in the mask = one soft body
///   - Black (or transparent) = no body (background, will be skipped)
///   - Must be the same aspect ratio as the color image
///
/// SETUP: Add to the same GameObject as ImageToFluid and FluidSimulationJobs.
///        Assign the mask texture in the inspector.
///        After ImageToFluid processes the color image, this component:
///          1. Samples the mask at the same resolution
///          2. Assigns a bodyIndex to each particle
///          3. Creates springs between neighboring particles of the same body
///
/// If no mask is assigned, all particles are treated as one body (bodyIndex = 0).
/// </summary>
public class SoftBodySetup : MonoBehaviour
{
    [Header("Mask")]
    [Tooltip("Mask texture where each color defines a separate soft body. " +
             "Black/transparent = background. Must have Read/Write enabled.")]
    public Texture2D maskTexture;

    [Header("Spring Properties")]
    [Tooltip("How many neighbor rings to connect. 1 = immediate neighbors (4-8 springs per particle). " +
             "2 = two rings deep (12-20 springs, more rigid).")]
    [Range(1, 2)]
    public int connectionRings = 1;

    [Tooltip("Default break threshold: spring breaks when stretched beyond restLength × this value. " +
             "2.0 = breaks at 2x original length. Higher = harder to tear.")]
    [Range(1.5f, 5f)]
    public float defaultBreakThreshold = 2.5f;

    [Tooltip("Color distance threshold for considering two mask pixels as the same body")]
    [Range(0.01f, 0.3f)]
    public float maskColorThreshold = 0.1f;

    // ─── Output Data ─────────────────────────────────────────────

    /// <summary>True after processing is complete.</summary>
    public bool IsReady { get; private set; }

    /// <summary>Per-particle body index. -1 = no body (background).</summary>
    public int[] BodyIndices { get; private set; }

    /// <summary>All springs connecting particles within bodies.</summary>
    public SoftBodySpring[] Springs { get; private set; }

    /// <summary>Number of springs created.</summary>
    public int SpringCount { get; private set; }

    /// <summary>Number of distinct bodies found in the mask.</summary>
    public int BodyCount { get; private set; }

    /// <summary>Number of particles per body.</summary>
    public int[] ParticlesPerBody { get; private set; }

    /// <summary>Representative color for each body (from mask).</summary>
    public Color[] BodyColors { get; private set; }

    // ─── Internal ────────────────────────────────────────────────

    // Grid dimensions matching ImageToFluid sampling
    private int sampleW, sampleH;

    // ─── Public API ──────────────────────────────────────────────

    /// <summary>
    /// Call after ImageToFluid has processed its image.
    /// Reads the mask at the same resolution and builds spring connections.
    /// </summary>
    public void Process(ImageToFluid imageSource)
    {
        if (imageSource == null || !imageSource.IsReady)
        {
            Debug.LogWarning("[SoftBodySetup] ImageToFluid not ready, treating all as one body.");
            CreateSingleBody(imageSource);
            return;
        }

        if (maskTexture == null)
        {
            Debug.Log("[SoftBodySetup] No mask texture — all particles are one body.");
            CreateSingleBody(imageSource);
            return;
        }

        int particleCount = imageSource.GeneratedParticleCount;
        var particles = imageSource.GeneratedParticles;
        float spacing = imageSource.ComputedSpacing;

        // ── Step 1: Sample mask at the same resolution as the color image ──
        int imgW = maskTexture.width;
        int imgH = maskTexture.height;

        // Reconstruct sample dimensions from ImageToFluid's logic
        float aspect = (float)imgW / imgH;
        int resolution = 80; // Match ImageToFluid resolution — ideally read from it

        // Try to read resolution from ImageToFluid
        var itfResolution = imageSource.resolution;
        if (itfResolution > 0) resolution = itfResolution;

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

        Color[] maskPixels;
        try
        {
            maskPixels = maskTexture.GetPixels();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[SoftBodySetup] Failed to read mask: {e.Message}");
            CreateSingleBody(imageSource);
            return;
        }

        // Sample mask at particle resolution
        Color[] sampledMask = new Color[sampleW * sampleH];
        for (int y = 0; y < sampleH; y++)
        {
            for (int x = 0; x < sampleW; x++)
            {
                float u = (x + 0.5f) / sampleW;
                float v = (y + 0.5f) / sampleH;
                int ix = Mathf.Clamp(Mathf.FloorToInt(u * imgW), 0, imgW - 1);
                int iy = Mathf.Clamp(Mathf.FloorToInt(v * imgH), 0, imgH - 1);
                sampledMask[y * sampleW + x] = maskPixels[iy * imgW + ix];
            }
        }

        // ── Step 2: Identify unique body colors in the mask ──
        List<Color> bodyColorList = new List<Color>();
        int[] maskBodyMap = new int[sampleW * sampleH]; // per-sample-pixel body index

        for (int i = 0; i < sampledMask.Length; i++)
        {
            Color c = sampledMask[i];

            // Black or transparent = background
            if (c.a < 0.1f || (c.r < 0.05f && c.g < 0.05f && c.b < 0.05f))
            {
                maskBodyMap[i] = -1;
                continue;
            }

            // Find matching existing body color
            int bodyIdx = -1;
            for (int b = 0; b < bodyColorList.Count; b++)
            {
                float dr = c.r - bodyColorList[b].r;
                float dg = c.g - bodyColorList[b].g;
                float db = c.b - bodyColorList[b].b;
                if (dr * dr + dg * dg + db * db < maskColorThreshold * maskColorThreshold)
                {
                    bodyIdx = b;
                    break;
                }
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

        Debug.Log($"[SoftBodySetup] Found {BodyCount} bodies in mask");

        // ── Step 3: Map particles to body indices ──
        // ImageToFluid creates particles by iterating y,x and skipping transparent pixels.
        // We need to match this iteration order.
        BodyIndices = new int[particleCount];
        ParticlesPerBody = new int[BodyCount];

        int pIdx = 0;

        // We need to read the color image's sampled pixels to know which were skipped.
        // Since ImageToFluid already processed, we reconstruct the mapping:
        // Particle i corresponds to the i-th non-transparent pixel in row-major order.
        Color[] colorPixels;
        try
        {
            colorPixels = imageSource.sourceImage.GetPixels();
        }
        catch
        {
            Debug.LogError("[SoftBodySetup] Failed to read color image for mapping.");
            CreateSingleBody(imageSource);
            return;
        }

        // Sample color at same resolution
        for (int y = 0; y < sampleH; y++)
        {
            for (int x = 0; x < sampleW; x++)
            {
                float u = (x + 0.5f) / sampleW;
                float v = (y + 0.5f) / sampleH;
                int ix = Mathf.Clamp(Mathf.FloorToInt(u * imageSource.sourceImage.width), 0, imageSource.sourceImage.width - 1);
                int iy = Mathf.Clamp(Mathf.FloorToInt(v * imageSource.sourceImage.height), 0, imageSource.sourceImage.height - 1);

                Color colorPixel = colorPixels[iy * imageSource.sourceImage.width + ix];

                // Skip transparent pixels (matches ImageToFluid logic)
                if (colorPixel.a <= 0.1f) continue;

                if (pIdx >= particleCount) break;

                int maskSampleIdx = y * sampleW + x;
                int bodyIdx = maskBodyMap[maskSampleIdx];

                // If mask says background but color image has a pixel, assign to body 0
                if (bodyIdx < 0 && BodyCount > 0) bodyIdx = 0;
                if (bodyIdx < 0) bodyIdx = 0;

                BodyIndices[pIdx] = bodyIdx;
                if (bodyIdx >= 0 && bodyIdx < BodyCount)
                    ParticlesPerBody[bodyIdx]++;

                pIdx++;
            }
        }

        Debug.Log($"[SoftBodySetup] Mapped {pIdx} particles to bodies");
        for (int b = 0; b < BodyCount; b++)
            Debug.Log($"  Body {b}: {ParticlesPerBody[b]} particles, " +
                      $"color=#{ColorUtility.ToHtmlStringRGB(BodyColors[b])}");

        // ── Step 4: Create springs between neighboring particles ──
        CreateSprings(particles, spacing);

        IsReady = true;
    }

    // ─── Spring Generation ───────────────────────────────────────

    /// <summary>
    /// Creates springs between nearby particles that belong to the same body.
    /// Uses a spatial approach: for each particle, checks neighbors within
    /// connectionRings × spacing distance.
    /// </summary>
    void CreateSprings(FluidParticle[] particles, float spacing)
    {
        int particleCount = particles.Length;
        float maxDist = spacing * (connectionRings + 0.5f); // Slightly more than grid distance
        float maxDistSqr = maxDist * maxDist;

        // Build a quick spatial lookup (reuse similar logic to spatial hash)
        // For init only, so O(n²) within local neighborhood is fine
        var springList = new List<SoftBodySpring>();

        // To avoid duplicate springs (A→B and B→A), only create if i < j
        // Use a spatial grid for efficiency
        float cellSize = maxDist;
        float minX = float.MaxValue, minY = float.MaxValue;
        for (int i = 0; i < particleCount; i++)
        {
            if (particles[i].position.x < minX) minX = particles[i].position.x;
            if (particles[i].position.y < minY) minY = particles[i].position.y;
        }

        var gridDict = new Dictionary<int, List<int>>();
        int gridW = 1000; // Hash grid width for init

        for (int i = 0; i < particleCount; i++)
        {
            int cx = Mathf.FloorToInt((particles[i].position.x - minX) / cellSize);
            int cy = Mathf.FloorToInt((particles[i].position.y - minY) / cellSize);
            int key = cy * gridW + cx;

            if (!gridDict.ContainsKey(key))
                gridDict[key] = new List<int>();
            gridDict[key].Add(i);
        }

        for (int i = 0; i < particleCount; i++)
        {
            if (BodyIndices[i] < 0) continue;

            int cx = Mathf.FloorToInt((particles[i].position.x - minX) / cellSize);
            int cy = Mathf.FloorToInt((particles[i].position.y - minY) / cellSize);

            // Check 3x3 neighborhood
            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                int key = (cy + dy) * gridW + (cx + dx);
                if (!gridDict.TryGetValue(key, out var cell)) continue;

                for (int n = 0; n < cell.Count; n++)
                {
                    int j = cell[n];
                    if (j <= i) continue; // Avoid duplicates

                    // Must be same body
                    if (BodyIndices[j] != BodyIndices[i]) continue;

                    Vector2 diff = particles[i].position - particles[j].position;
                    float distSqr = diff.sqrMagnitude;

                    if (distSqr < maxDistSqr && distSqr > 0.0001f)
                    {
                        float dist = Mathf.Sqrt(distSqr);

                        springList.Add(new SoftBodySpring
                        {
                            particleA = i,
                            particleB = j,
                            restLength = dist,
                            breakThreshold = defaultBreakThreshold,
                            alive = 1
                        });
                    }
                }
            }
        }

        Springs = springList.ToArray();
        SpringCount = Springs.Length;

        // Stats
        float avgSpringsPerParticle = particleCount > 0
            ? (float)SpringCount * 2f / particleCount : 0f;

        Debug.Log($"[SoftBodySetup] Created {SpringCount} springs " +
                  $"(avg {avgSpringsPerParticle:F1} per particle, " +
                  $"rings={connectionRings}, breakThreshold={defaultBreakThreshold:F1})");
    }

    // ─── Fallback: Single Body ───────────────────────────────────

    /// <summary>
    /// When no mask is provided, all particles belong to body 0.
    /// Springs are still created between neighbors.
    /// </summary>
    void CreateSingleBody(ImageToFluid imageSource)
    {
        if (imageSource == null || !imageSource.IsReady)
        {
            BodyIndices = new int[0];
            Springs = new SoftBodySpring[0];
            SpringCount = 0;
            BodyCount = 0;
            IsReady = true;
            return;
        }

        int particleCount = imageSource.GeneratedParticleCount;
        var particles = imageSource.GeneratedParticles;
        float spacing = imageSource.ComputedSpacing;

        BodyCount = 1;
        BodyColors = new Color[] { Color.white };
        BodyIndices = new int[particleCount];
        ParticlesPerBody = new int[] { particleCount };

        for (int i = 0; i < particleCount; i++)
            BodyIndices[i] = 0;

        CreateSprings(particles, spacing);
        IsReady = true;
    }
}