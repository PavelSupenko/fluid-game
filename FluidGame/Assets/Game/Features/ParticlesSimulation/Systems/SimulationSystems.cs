using ParticlesSimulation.Components;
using Unity.Collections;
using Unity.Jobs;
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
                BoundaryFriction = math.saturate(config.boundaryFriction),
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
    /// XSPH velocity smoothing system. Blends each particle's velocity toward its
    /// neighbors' average using the Poly6 kernel, producing cohesive viscous flow.
    /// Runs after finalization computes velocities, using the spatial hash grid
    /// built earlier in the frame (still valid — particles moved only small amounts).
    /// Pipeline: … → Finalization → <b>XSPH Viscosity</b> → LocalTransformSync.
    /// </summary>
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(FinalizationSystem))]
    public partial class XsphViscositySystem : SystemBase
    {
        private SpatialHashGridSystem _spatialHashSystem;
        private EntityQuery _particleQuery;

        private NativeArray<float2> _velocities;
        private NativeArray<float2> _positions;
        private NativeArray<float> _densities;
        private NativeArray<float2> _smoothedVelocities;
        private bool _isAllocated;

        protected override void OnCreate()
        {
            _spatialHashSystem = World.GetExistingSystemManaged<SpatialHashGridSystem>();

            // Query must match SpatialHashGridSystem and PbfSolverSystem exactly
            // so that EntityIndexInQuery produces identical particle ordering.
            _particleQuery = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleFluid, ParticleState, ParticleSimulatedTag>()
                .Build();

            RequireForUpdate(_particleQuery);
            RequireForUpdate<SimulationConfig>();
        }

        protected override void OnUpdate()
        {
            var config = SystemAPI.GetSingleton<SimulationConfig>();
            if (config.xsphViscosity <= 0f)
                return;

            var particleCount = _spatialHashSystem.ParticleCount;
            if (particleCount == 0)
                return;

            EnsureCapacity(particleCount, config.maxParticles);

            var handle = JobHandle.CombineDependencies(Dependency, _spatialHashSystem.FinalJobHandle);

            // Extract current velocities, positions, and densities into flat arrays.
            var velocities = _velocities.GetSubArray(0, particleCount);
            var positions = _positions.GetSubArray(0, particleCount);
            var densities = _densities.GetSubArray(0, particleCount);
            var smoothed = _smoothedVelocities.GetSubArray(0, particleCount);

            handle = new ExtractParticleDataJob
            {
                Velocities = velocities,
                Positions = positions,
                Densities = densities
            }.ScheduleParallel(_particleQuery, handle);

            handle = new Jobs.XsphViscosityJob
            {
                Positions = positions,
                Velocities = velocities,
                Densities = densities,
                Grid = _spatialHashSystem.Grid,
                CellSizeInverse = config.cellSizeInv,
                SmoothingRadiusSq = config.smoothingRadiusSq,
                Poly6Coefficient = config.poly6Coefficient,
                Viscosity = config.xsphViscosity,
                SmoothedVelocities = smoothed
            }.Schedule(particleCount, 64, handle);

            handle = new Jobs.WriteBackXsphVelocitiesJob
            {
                SmoothedVelocities = smoothed
            }.ScheduleParallel(_particleQuery, handle);

            Dependency = handle;
        }

        /// <summary>
        /// Extracts velocity, position, and density from ECS into flat NativeArrays
        /// for the XSPH job (which needs random-access by neighbor index).
        /// </summary>
        [BurstCompile]
        [WithAll(typeof(ParticleSimulatedTag))]
        private partial struct ExtractParticleDataJob : IJobEntity
        {
            [WriteOnly] public NativeArray<float2> Velocities;
            [WriteOnly] public NativeArray<float2> Positions;
            [WriteOnly] public NativeArray<float> Densities;

            public void Execute([EntityIndexInQuery] int index, in ParticleCore core, in ParticleFluid fluid)
            {
                Velocities[index] = core.velocity;
                Positions[index] = core.position;
                Densities[index] = fluid.density;
            }
        }

        private void EnsureCapacity(int particleCount, int maxParticles)
        {
            var required = math.max(particleCount, maxParticles);

            if (_isAllocated && _velocities.Length >= required)
                return;

            if (_isAllocated)
            {
                _velocities.Dispose();
                _positions.Dispose();
                _densities.Dispose();
                _smoothedVelocities.Dispose();
            }

            _velocities = new NativeArray<float2>(required, Allocator.Persistent);
            _positions = new NativeArray<float2>(required, Allocator.Persistent);
            _densities = new NativeArray<float>(required, Allocator.Persistent);
            _smoothedVelocities = new NativeArray<float2>(required, Allocator.Persistent);
            _isAllocated = true;
        }

        protected override void OnDestroy()
        {
            if (!_isAllocated)
                return;

            _velocities.Dispose();
            _positions.Dispose();
            _densities.Dispose();
            _smoothedVelocities.Dispose();
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