using Unity.Entities;
using Unity.Mathematics;

namespace ParticlesSimulation.Components
{
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
        public float deltaScale;
        /// <summary>Neighbor kernel mass in density/lambda jobs (must match per-particle mass if uniform).</summary>
        public float uniformParticleMass;
    }

    public static class SimulationConfigUtility
    {
        public static SimulationConfig CreateDefault(int maxParticles)
        {
            const float h = 0.12f;
            var h2 = h * h;
            var h8 = h2 * h2 * h2 * h2;
            var poly6 = 4f / (math.PI * h8);
            var h5 = h * h * h * h * h;
            var spiky = -10f / (math.PI * h5);

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
                spikyGradCoefficient = spiky,
                solverIterations = 2,
                maxParticles = math.max(1024, maxParticles),
                pbfEpsilon = 120f,
                deltaScale = 1f,
                uniformParticleMass = 1f
            };
        }

        public static void ApplySmoothingRadius(ref SimulationConfig c, float h)
        {
            var h2 = h * h;
            var h8 = h2 * h2 * h2 * h2;
            var poly6 = 4f / (math.PI * h8);
            var h5 = h * h * h * h * h;
            var spiky = -10f / (math.PI * h5);
            c.smoothingRadius = h;
            c.smoothingRadiusSq = h2;
            c.cellSizeInv = 1f / h;
            c.poly6Coefficient = poly6;
            c.spikyGradCoefficient = spiky;
        }
    }
}
