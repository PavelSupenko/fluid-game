using Unity.Entities;
using Unity.Mathematics;
using UnityEngine;

namespace ParticlesSimulation
{
    /// <summary>
    /// Keeps <see cref="SimulationConfig.deltaTime"/> aligned with the frame (clamped for stability).
    /// </summary>
    [UpdateInGroup(typeof(ParticleSimulationGroup), OrderFirst = true)]
    public partial struct ParticleSimulationClockSystem : ISystem
    {
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<SimulationConfig>();
        }

        public void OnUpdate(ref SystemState state)
        {
            var cfg = SystemAPI.GetSingletonRW<SimulationConfig>();
            cfg.ValueRW.deltaTime = math.min(Time.deltaTime, 1f / 20f);
        }
    }
}
