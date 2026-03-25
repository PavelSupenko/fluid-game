using Unity.Burst;
using Unity.Entities;

namespace ParticlesSimulation
{
    /// <summary>
    /// Switches particles to fluid when world <see cref="ParticleCore.position"/>.y is at or below MeltLineY.
    /// </summary>
    [BurstCompile]
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(PredictPositionsSystem))]
    public partial struct MeltingTriggerSystem : ISystem
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
            var meltY = SystemAPI.GetSingleton<SimulationConfig>().meltLineY;

            foreach (var (core, st) in SystemAPI.Query<RefRO<ParticleCore>, RefRW<ParticleState>>().WithAll<ParticleSimTag>())
            {
                if (st.ValueRO.phase != ParticlePhase.Rigid)
                    continue;
                if (core.ValueRO.position.y > meltY)
                    continue;
                st.ValueRW.phase = ParticlePhase.Fluid;
            }
        }
    }
}
