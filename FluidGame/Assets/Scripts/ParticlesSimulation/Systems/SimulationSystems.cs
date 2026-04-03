using ParticlesSimulation.Components;
using Unity.Mathematics;
using Unity.Transforms;
using Unity.Entities;
using Unity.Burst;

namespace ParticlesSimulation.Systems
{
    /// <summary>
    /// Keeps <see cref="SimulationConfig.deltaTime"/> at a fixed simulation step (1/60s).
    /// At framerates below 60fps the simulation slows down proportionally rather than
    /// taking larger timesteps, which preserves stability. A full substep approach
    /// (running the solver multiple times per frame) can be added later if needed.
    /// </summary>
    [UpdateInGroup(typeof(ParticleSimulationGroup), OrderFirst = true)]
    public partial struct ParticleSimulationClockSystem : ISystem
    {
        private const float FixedSimDt = 1f / 60f;

        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<SimulationConfig>();
        }

        public void OnUpdate(ref SystemState state)
        {
            var cfg = SystemAPI.GetSingletonRW<SimulationConfig>();
            cfg.ValueRW.deltaTime = FixedSimDt;
        }
    }

    /// <summary>
    /// Applies external forces (gravity) and computes predicted positions for the solver.
    /// Pipeline: Clock → <b>Prediction</b> → [future PBF solver] → Finalization.
    /// </summary>
    [BurstCompile]
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(ParticleSimulationClockSystem))]
    public partial struct PredictionSystem : ISystem
    {
        private EntityQuery _query;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            _query = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleState, ParticleSimulatedTag>()
                .Build();
            state.RequireForUpdate(_query);
            state.RequireForUpdate<SimulationConfig>();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var cfg = SystemAPI.GetSingleton<SimulationConfig>();

            state.Dependency = new Jobs.PredictPositionsJob
            {
                DeltaTime = cfg.deltaTime,
                Gravity = new float2(0f, cfg.gravityY)
            }.ScheduleParallel(_query, state.Dependency);
        }
    }

    /// <summary>
    /// Finalizes velocity from constrained predictions, commits positions, then applies light fluid damping.
    /// Pipeline: Clock → Prediction → SpatialHash → PBF Solver → <b>Finalization</b>.
    /// </summary>
    [BurstCompile]
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(PbfSolverSystem))]
    public partial struct FinalizationSystem : ISystem
    {
        private EntityQuery _query;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            _query = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleState, ParticleSimulatedTag>()
                .Build();
            state.RequireForUpdate(_query);
            state.RequireForUpdate<SimulationConfig>();
            state.RequireForUpdate<SimulationWorldBounds>();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var config = SystemAPI.GetSingleton<SimulationConfig>();
            var inverseDeltaTime = math.rcp(config.deltaTime);
            var bounds = SystemAPI.GetSingleton<SimulationWorldBounds>();

            var handle = state.Dependency;

            var maxDisplacement = config.maxDisplacementFraction * config.smoothingRadius;

            handle = new Jobs.FinalizePositionsJob
            {
                InverseDeltaTime = inverseDeltaTime,
                MaxSpeed = config.maxSpeed,
                MaxSpeedSq = config.maxSpeed * config.maxSpeed,
                MaxDisplacement = maxDisplacement,
                MaxDisplacementSq = maxDisplacement * maxDisplacement,
                WorldBounds = bounds
            }.ScheduleParallel(_query, handle);

            // fluidDamping is applied directly as a per-frame fraction (not scaled by dt).
            // For thick viscous fluids, values of 0.2–0.5 give heavy, honey-like behavior.
            handle = new Jobs.ApplyScalarFluidDampingJob
            {
                Damping = math.saturate(config.fluidDamping)
            }.ScheduleParallel(_query, handle);

            state.Dependency = handle;
        }
    }

    /// <summary>
    /// Keeps <see cref="LocalTransform"/> aligned with simulation <see cref="ParticleCore"/> for Entities Graphics.
    /// Runs after the full simulation group commits positions.
    /// </summary>
    [BurstCompile]
    [UpdateInGroup(typeof(SimulationSystemGroup))]
    [UpdateAfter(typeof(ParticleSimulationGroup))]
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
            foreach (var (core, lt) in SystemAPI
                         .Query<RefRO<ParticleCore>, RefRW<LocalTransform>>()
                         .WithAll<ParticleSimulatedTag>())
            {
                float2 p = core.ValueRO.position;
                lt.ValueRW.Position = new float3(p.x, p.y, 0f);
            }
        }
    }
}