using ParticlesSimulation.Components;
using Unity.Burst;
using Unity.Entities;
using Unity.Mathematics;

namespace ParticlesSimulation.Systems
{
    /// <summary>
    /// Applies gravity and integrates predicted positions from velocity (semi-implicit Euler prediction).
    /// </summary>
    [BurstCompile]
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(ParticleSimulationClockSystem))]
    public partial struct PredictPositionsSystem : ISystem
    {
        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<SimulationConfig>();
            state.RequireForUpdate<ParticleSimTag>();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var cfg = SystemAPI.GetSingleton<SimulationConfig>();
            var dt = cfg.deltaTime;
            var gravity = new float2(0f, cfg.gravityY);

            foreach (var core in SystemAPI.Query<RefRW<ParticleCore>>().WithAll<ParticleSimTag>())
            {
                var c = core.ValueRO;
                var vAdj = c.velocity + gravity * dt;
                var predicted = c.position + vAdj * dt;
                core.ValueRW.predictedPosition = predicted;
            }
        }
    }
}
