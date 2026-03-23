using UnityEngine;

/// <summary>
/// Standard SPH smoothing kernel functions for 2D simulation.
/// 
/// Poly6 kernel  — used for density estimation (smooth, non-zero at center).
/// Spiky kernel  — used for pressure forces (sharp gradient at small distances,
///                 prevents particles from clumping at the same position).
/// 
/// All kernels are parameterized by the smoothing radius h.
/// Normalization constants are for 2D (not 3D).
/// </summary>
public static class SPHKernels
{
    // ─── Poly6 Kernel (density) ──────────────────────────────────

    /// <summary>
    /// Precomputes the normalization constant for the 2D Poly6 kernel.
    /// Call once when h changes. Formula: 4 / (π * h^8)
    /// </summary>
    public static float Poly6Coefficient(float h)
    {
        float h2 = h * h;
        float h8 = h2 * h2 * h2 * h2;
        return 4f / (Mathf.PI * h8);
    }

    /// <summary>
    /// Evaluates the Poly6 kernel for a given squared distance.
    /// Uses r² directly to avoid computing a square root.
    /// Returns 0 if r² >= h².
    /// </summary>
    public static float Poly6(float rSqr, float hSqr, float coefficient)
    {
        if (rSqr >= hSqr) return 0f;

        float diff = hSqr - rSqr;
        return coefficient * diff * diff * diff;
    }

    // ─── Spiky Kernel Gradient (pressure) ────────────────────────

    /// <summary>
    /// Precomputes the normalization constant for the 2D Spiky gradient kernel.
    /// Call once when h changes. Formula: -10 / (π * h^5)
    /// </summary>
    public static float SpikyGradCoefficient(float h)
    {
        float h5 = h * h * h * h * h;
        return -10f / (Mathf.PI * h5);
    }

    /// <summary>
    /// Evaluates the Spiky kernel gradient magnitude for a given distance.
    /// Returns a scalar; caller multiplies by the normalized direction vector.
    /// Returns 0 if r >= h or r is near zero.
    /// </summary>
    public static float SpikyGrad(float r, float h, float coefficient)
    {
        if (r >= h || r < 1e-6f) return 0f;

        float diff = h - r;
        return coefficient * diff * diff;
    }
}
