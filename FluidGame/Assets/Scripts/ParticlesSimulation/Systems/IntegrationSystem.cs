using ParticlesSimulation.Components;
using Unity.Entities;
using Unity.Mathematics;

namespace ParticlesSimulation.Systems
{
    /// <summary>
    /// Finalizes velocity from constrained predictions, commits positions, then applies light fluid damping.
    /// Neighbor-based XSPH is omitted here because parallel ComponentLookup&lt;ParticleCore&gt; reads alias with
    /// ref ParticleCore writes under Entities safety rules; reintroduce via a double-buffer or IJobChunk if needed.
    /// </summary>
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(ParticleSimulationClockSystem))]
    public partial struct IntegrationSystem : ISystem
    {
        private EntityQuery query;

        public void OnCreate(ref SystemState state)
        {
            query = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleFluid, ParticleState, GridHash, ParticleSimTag>()
                .Build();
            state.RequireForUpdate(query);
            state.RequireForUpdate<SimulationConfig>();
            state.RequireForUpdate<SimulationWorldBounds>();
        }

        public void OnUpdate(ref SystemState state)
        {
            var cfg = SystemAPI.GetSingleton<SimulationConfig>();
            var invDt = math.rcp(cfg.deltaTime);
            var bounds = SystemAPI.GetSingleton<SimulationWorldBounds>();

            var handle = state.Dependency;

            handle = new Jobs.IntegratePositionsJob { invDt = invDt, WorldBounds = bounds }.ScheduleParallel(query, handle);

            handle = new Jobs.ApplyScalarFluidDampingJob
            {
                damping = math.saturate(cfg.viscosityMultiplier * cfg.deltaTime)
            }.ScheduleParallel(query, handle);

            state.Dependency = handle;
        }
    }
}
