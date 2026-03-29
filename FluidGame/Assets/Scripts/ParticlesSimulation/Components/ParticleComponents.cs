using Unity.Entities;
using Unity.Mathematics;

namespace ParticlesSimulation.Components
{
    /// <summary>
    /// Core kinematic state (2D simulation uses xy; z unused).
    /// </summary>
    public struct ParticleCore : IComponentData
    {
        public float2 position;
        public float2 predictedPosition;
        public float2 velocity;
        public float mass;
    }

    /// <summary>
    /// Fluid samples: density/pressure each frame; rest density and mass are constants.
    /// </summary>
    public struct ParticleFluid : IComponentData
    {
        public float density;
        public float pressure;
        public float restDensity;
        public float mass;
        /// <summary>PBF Lagrange multiplier scratch for the current solver pass.</summary>
        public float lambda;
    }

    /// <summary>
    /// Phase, palette id, and rest pose for rigid shape matching (local to initial COM).
    /// </summary>
    public struct ParticleState : IComponentData
    {
        public ParticlePhase phase;
        public float2 initialLocalPosition;
        public byte colorId;
    }

    /// <summary>
    /// Cached 1D spatial hash for the particle's cell (debug / optional use).
    /// </summary>
    public struct GridHash : IComponentData
    {
        public int cellHash;
    }

    /// <summary>
    /// Linear sRGB color for the simple instanced quad path.
    /// </summary>
    public struct ParticleDrawColor : IComponentData
    {
        public float4 value;
    }

    /// <summary>
    /// Tags the particle simulation archetype for queries.
    /// </summary>
    public struct ParticleSimTag : IComponentData
    {
    }

    /// <summary>
    /// Singleton tag for simulation bootstrap / bounds sync (spatial hash buffers live on <see cref="ParticlesSimulation.Systems.ParticlePbfLoopSystem"/>).
    /// </summary>
    public struct SpatialGridMapTag : IComponentData
    {
    }

    /// <summary>
    /// Latest rigid center of mass in world space (xy). Count is number of rigid particles.
    /// </summary>
    public struct RigidComState : IComponentData
    {
        public float2 center;
        public int count;
    }
}
