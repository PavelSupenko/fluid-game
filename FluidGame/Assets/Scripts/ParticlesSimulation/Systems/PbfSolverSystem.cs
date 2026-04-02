using ParticlesSimulation.Components;
using ParticlesSimulation.Jobs;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Systems
{
    /// <summary>
    /// Position Based Fluids solver. Computes density and Lagrange multiplier (λ)
    /// for all particles using the spatial hash grid built by <see cref="SpatialHashGridSystem"/>.
    /// Results are written back into <see cref="ParticleFluid"/> components.
    /// Pipeline: Clock → Prediction → SpatialHash → <b>PBF Solver</b> → Finalization.
    /// </summary>
    /// <remarks>
    /// Step 3: density + lambda only (single pass, no position correction yet).
    /// Step 4 will wrap the density→lambda→deltaP cycle in a solver iteration loop.
    /// </remarks>
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(SpatialHashGridSystem))]
    [UpdateBefore(typeof(FinalizationSystem))]
    public partial class PbfSolverSystem : SystemBase
    {
        private SpatialHashGridSystem _spatialHashSystem;
        private EntityQuery _particleQuery;

        private NativeArray<float> _densities;
        private NativeArray<float> _lambdas;
        private bool _isAllocated;

        protected override void OnCreate()
        {
            _spatialHashSystem = World.GetExistingSystemManaged<SpatialHashGridSystem>();

            _particleQuery = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleFluid, ParticleState, ParticleSimulatedTag>()
                .Build();

            RequireForUpdate(_particleQuery);
            RequireForUpdate<SimulationConfig>();
        }

        protected override void OnUpdate()
        {
            var config = SystemAPI.GetSingleton<SimulationConfig>();
            var particleCount = _spatialHashSystem.ParticleCount;

            if (particleCount == 0)
                return;

            EnsureCapacity(particleCount, config.maxParticles);

            // Chain dependency with spatial hash system to safely read its native containers.
            var handle = JobHandle.CombineDependencies(Dependency, _spatialHashSystem.FinalJobHandle);

            var positions = _spatialHashSystem.Positions.GetSubArray(0, particleCount);
            var grid = _spatialHashSystem.Grid;
            var densitySlice = _densities.GetSubArray(0, particleCount);
            var lambdaSlice = _lambdas.GetSubArray(0, particleCount);

            // --- Density ---
            handle = new ComputeDensityJob
            {
                Positions = positions,
                Grid = grid,
                CellSizeInverse = config.cellSizeInv,
                SmoothingRadiusSq = config.smoothingRadiusSq,
                Poly6Coefficient = config.poly6Coefficient,
                ParticleMass = config.uniformParticleMass,
                Densities = densitySlice
            }.Schedule(particleCount, 64, handle);

            // --- Lambda ---
            handle = new ComputeLambdaJob
            {
                Positions = positions,
                Densities = densitySlice,
                Grid = grid,
                CellSizeInverse = config.cellSizeInv,
                SmoothingRadius = config.smoothingRadius,
                SmoothingRadiusSq = config.smoothingRadiusSq,
                SpikyGradCoefficient = config.spikyGradCoefficient,
                RestDensity = config.restDensity,
                ParticleMass = config.uniformParticleMass,
                Epsilon = config.pbfEpsilon,
                Lambdas = lambdaSlice
            }.Schedule(particleCount, 64, handle);

            // --- Write back to ECS components ---
            handle = new WriteBackFluidDataJob
            {
                Densities = densitySlice,
                Lambdas = lambdaSlice
            }.ScheduleParallel(_particleQuery, handle);

            Dependency = handle;
        }

        private void EnsureCapacity(int particleCount, int maxParticles)
        {
            var requiredCapacity = math.max(particleCount, maxParticles);

            if (_isAllocated && _densities.Length >= requiredCapacity)
                return;

            if (_isAllocated)
            {
                _densities.Dispose();
                _lambdas.Dispose();
            }

            _densities = new NativeArray<float>(requiredCapacity, Allocator.Persistent);
            _lambdas = new NativeArray<float>(requiredCapacity, Allocator.Persistent);
            _isAllocated = true;
        }

        protected override void OnDestroy()
        {
            if (!_isAllocated)
                return;

            _densities.Dispose();
            _lambdas.Dispose();
        }
    }
}
