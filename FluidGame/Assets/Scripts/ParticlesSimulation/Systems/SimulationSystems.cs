using ParticlesSimulation.Components;
using Unity.Mathematics;
using Unity.Transforms;
using Unity.Entities;
using Unity.Burst;

namespace ParticlesSimulation.Systems
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
            cfg.ValueRW.deltaTime = math.min(SystemAPI.Time.DeltaTime, 1f / 20f);
        }
    }
    
    /// <summary>
    /// Keeps <see cref="LocalTransform"/> aligned with simulation <see cref="ParticleCore"/> for Entities Graphics.
    /// </summary>
    [BurstCompile]
    [UpdateInGroup(typeof(SimulationSystemGroup))]
    [UpdateAfter(typeof(IntegrationSystem))]
    [UpdateBefore(typeof(TransformSystemGroup))]
    public partial struct ParticleLocalTransformSyncSystem : ISystem
    {
        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<ParticleSimulatedTag>();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            foreach (var (core, lt) in SystemAPI.Query<RefRO<ParticleCore>, RefRW<LocalTransform>>().WithAll<ParticleSimulatedTag>())
            {
                float2 p = core.ValueRO.position;
                lt.ValueRW.Position = new float3(p.x, p.y, 0f);
            }
        }
    }

    
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
                .WithAll<ParticleCore, ParticleFluid, ParticleState, ParticleSimulatedTag>()
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

            handle = new Jobs.FinalizePositionsJob { invDt = invDt, WorldBounds = bounds }.ScheduleParallel(query, handle);

            handle = new Jobs.ApplyScalarFluidDampingJob
            {
                damping = math.saturate(cfg.viscosityMultiplier * cfg.deltaTime)
            }.ScheduleParallel(query, handle);

            state.Dependency = handle;
        }
    }
}
