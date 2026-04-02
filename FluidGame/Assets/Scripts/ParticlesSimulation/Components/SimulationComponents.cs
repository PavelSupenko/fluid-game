using Unity.Entities;
using Unity.Mathematics;

namespace ParticlesSimulation.Components
{
    /// <summary>
    /// Rigid particles preserve image structure via shape matching; fluid particles use PBF.
    /// </summary>
    public enum ParticlePhase : byte
    {
        Rigid = 0,
        Fluid = 1
    }
    
    /// <summary>
    /// Core kinematic state (2D simulation uses xy; z unused).
    /// Per-particle mass lives in <see cref="SimulationConfig.uniformParticleMass"/>.
    /// </summary>
    public struct ParticleCore : IComponentData
    {
        public float2 position;
        public float2 predictedPosition;
        public float2 velocity;
    }

    /// <summary>
    /// Fluid PBF state: density is recomputed each frame; rest density is a per-particle constant.
    /// Per-particle mass lives in <see cref="SimulationConfig.uniformParticleMass"/>.
    /// </summary>
    public struct ParticleFluid : IComponentData
    {
        public float density;
        public float restDensity;
        /// <summary>PBF Lagrange multiplier scratch for the current solver pass.</summary>
        public float lambda;
    }

    /// <summary>
    /// Phase tag and palette index. Rigid shape-matching data (rest pose, etc.)
    /// will be added as a separate component when that feature is implemented.
    /// </summary>
    public struct ParticleState : IComponentData
    {
        public ParticlePhase phase;
        public byte colorId;
    }

    /// <summary>
    /// Tags the particle simulation archetype for queries.
    /// </summary>
    public struct ParticleSimulatedTag : IComponentData
    {
    }

    /// <summary>
    /// Stores the original spawn/image color so debug visualization can restore it.
    /// Value is in linear color space (same as <see cref="Unity.Rendering.URPMaterialPropertyBaseColor"/>).
    /// </summary>
    public struct ParticleOriginalColor : IComponentData
    {
        public float4 Value;
    }

    /// <summary>
    /// Singleton tag for simulation bootstrap and bounds sync.
    /// </summary>
    public struct SpatialGridMapTag : IComponentData
    {
    }
    
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
    
    /// <summary>
    /// Global simulation parameters (singleton component on a dedicated entity).
    /// </summary>
    public struct SimulationConfig : IComponentData
    {
        public float gravityY;
        public float smoothingRadius;
        public float smoothingRadiusSq;
        public float cellSizeInv;
        public float meltLineY;
        public float viscosityMultiplier;
        public float stiffness;
        public float rigidShapeStiffness;
        public float deltaTime;
        public float poly6Coefficient;
        public float spikyGradCoefficient;
        public int solverIterations;
        public int maxParticles;
        public float pbfEpsilon;
        /// <summary>Rest density target for the PBF constraint (ρ/ρ₀ − 1 = 0).</summary>
        public float restDensity;
        /// <summary>Neighbor kernel mass in density/lambda jobs (must match per-particle mass if uniform).</summary>
        public float uniformParticleMass;
    }
    
    public static class ConfigUtility
    {
        public static SimulationConfig CreateDefault(int maxParticles)
        {
            const float h = 0.12f;
            var h2 = h * h;
            var h8 = h2 * h2 * h2 * h2;
            var poly6 = 4f / (math.PI * h8);
            var h5 = h * h * h * h * h;
            // 2D Spiky kernel gradient: ∇W = −(30/πh⁵)·(h−r)²·r̂
            var spikyGrad = -30f / (math.PI * h5);

            return new SimulationConfig
            {
                gravityY = -12f,
                smoothingRadius = h,
                smoothingRadiusSq = h2,
                cellSizeInv = 1f / h,
                meltLineY = -1.2f,
                viscosityMultiplier = 0.35f,
                stiffness = 0.8f,
                rigidShapeStiffness = 0.65f,
                deltaTime = 1f / 60f,
                poly6Coefficient = poly6,
                spikyGradCoefficient = spikyGrad,
                solverIterations = 2,
                maxParticles = math.max(1024, maxParticles),
                pbfEpsilon = 120f,
                restDensity = 300f,
                uniformParticleMass = 1f
            };
        }

        public static void ApplySmoothingRadius(ref SimulationConfig c, float h)
        {
            var h2 = h * h;
            var h8 = h2 * h2 * h2 * h2;
            var poly6 = 4f / (math.PI * h8);
            var h5 = h * h * h * h * h;
            var spikyGrad = -30f / (math.PI * h5);
            c.smoothingRadius = h;
            c.smoothingRadiusSq = h2;
            c.cellSizeInv = 1f / h;
            c.poly6Coefficient = poly6;
            c.spikyGradCoefficient = spikyGrad;
        }
    }
    
    /// <summary>
    /// Burst-compatible helpers for simulation boundary calculations.
    /// </summary>
    public static class BoundsUtility
    {
        /// <summary>
        /// Computes the effective margin clamped so it never collapses the box to zero.
        /// Shared between integration jobs and spawn-time inner-rect computation.
        /// </summary>
        public static float EffectiveMargin(float2 min, float2 max, float margin)
        {
            var ext = max - min;
            return math.min(margin, math.max(0f, 0.49f * math.cmin(ext)));
        }
    }
}