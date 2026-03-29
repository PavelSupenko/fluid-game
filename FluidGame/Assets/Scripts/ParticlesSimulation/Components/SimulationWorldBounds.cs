using Unity.Entities;
using Unity.Mathematics;

namespace ParticlesSimulation.Components
{
    /// <summary>
    /// Axis-aligned simulation region in world XY (singleton). When <see cref="BoundsEnabled"/> is non-zero,
    /// particle positions are clamped inside the padded box each integration step.
    /// </summary>
    public struct SimulationWorldBounds : IComponentData
    {
        /// <summary>0 = disabled, 1 = clamp particles to the box.</summary>
        public byte BoundsEnabled;
        public float2 Min;
        public float2 Max;
        /// <summary>Inward inset applied to Min/Max when clamping (world units).</summary>
        public float Margin;
    }
}
