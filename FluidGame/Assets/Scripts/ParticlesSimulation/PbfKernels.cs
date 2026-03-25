using Unity.Burst;
using Unity.Mathematics;

namespace ParticlesSimulation
{
    /// <summary>
    /// 2D Poly6 / Spiky kernels for PBF density and gradients. Inner loops use math.select to avoid branches.
    /// </summary>
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    public static class PbfKernels
    {
        [BurstCompile(FloatMode = FloatMode.Fast)]
        public static float Poly6(float r2, float h2, float coefficient)
        {
            var inside = math.select(0f, 1f, r2 < h2);
            var diff = h2 - r2;
            var diff3 = diff * diff * diff;
            return coefficient * diff3 * inside;
        }

        [BurstCompile(FloatMode = FloatMode.Fast)]
        public static float SpikyGradMagnitude(float r, float h, float coefficient)
        {
            var inside = math.select(0f, 1f, (r > 1e-8f) & (r < h));
            var diff = h - r;
            return coefficient * diff * diff * inside;
        }

        /// <summary>
        /// Normalized direction from i to j scaled by Spiky gradient magnitude (2D).
        /// </summary>
        [BurstCompile(FloatMode = FloatMode.Fast)]
        public static float2 SpikyGradVec(float2 delta, float h, float coefficient)
        {
            var r2 = math.lengthsq(delta);
            var r = math.sqrt(math.max(r2, 1e-16f));
            var mag = SpikyGradMagnitude(r, h, coefficient);
            var dir = delta * math.rcp(r);
            return dir * mag;
        }
    }
}
