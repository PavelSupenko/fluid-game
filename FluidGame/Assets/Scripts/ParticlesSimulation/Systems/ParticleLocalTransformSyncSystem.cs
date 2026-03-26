using ParticlesSimulation.Components;
using Unity.Burst;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;

namespace ParticlesSimulation.Systems
{
    /// <summary>
    /// Keeps <see cref="LocalTransform"/> aligned with simulation <see cref="ParticleCore"/> for Entities Graphics.
    /// </summary>
    [BurstCompile]
    [UpdateInGroup(typeof(SimulationSystemGroup))]
    [UpdateAfter(typeof(IntegrationSystem))]
    public partial struct ParticleLocalTransformSyncSystem : ISystem
    {
        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<ParticleSimTag>();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            foreach (var (core, lt) in SystemAPI.Query<RefRO<ParticleCore>, RefRW<LocalTransform>>().WithAll<ParticleSimTag>())
            {
                float2 p = core.ValueRO.position;
                lt.ValueRW.Position = new float3(p.x, p.y, 0f);
            }
        }
    }
}
