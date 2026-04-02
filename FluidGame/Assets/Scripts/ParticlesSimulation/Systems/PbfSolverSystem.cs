using ParticlesSimulation.Components;
using ParticlesSimulation.Jobs;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Systems
{
    /// <summary>
    /// Position Based Fluids solver. Iteratively resolves incompressibility constraints
    /// by computing density, Lagrange multiplier (λ), and position corrections (Δp).
    /// Pipeline: Clock → Prediction → SpatialHash → <b>PBF Solver</b> → Finalization.
    /// </summary>
    /// <remarks>
    /// The spatial hash grid is built once per frame (by <see cref="SpatialHashGridSystem"/>)
    /// and reused across solver iterations. This is standard practice for small iteration counts
    /// (2–4): corrections are small enough that particles don't cross cell boundaries.
    /// </remarks>
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(SpatialHashGridSystem))]
    [UpdateBefore(typeof(FinalizationSystem))]
    public partial class PbfSolverSystem : SystemBase
    {
        private SpatialHashGridSystem _spatialHashSystem;
        private EntityQuery _particleQuery;

        private NativeArray<float2> _workingPositions;
        private NativeArray<float> _densities;
        private NativeArray<float> _lambdas;
        private NativeArray<float2> _corrections;
        private bool _isAllocated;

        protected override void OnCreate()
        {
            _spatialHashSystem = World.GetExistingSystemManaged<SpatialHashGridSystem>();

            _particleQuery = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleFluid, ParticleState, ParticleSimulatedTag>()
                .Build();

            RequireForUpdate(_particleQuery);
            RequireForUpdate<SimulationConfig>();
            RequireForUpdate<SimulationWorldBounds>();
        }

        protected override void OnUpdate()
        {
            var config = SystemAPI.GetSingleton<SimulationConfig>();
            var bounds = SystemAPI.GetSingleton<SimulationWorldBounds>();
            var particleCount = _spatialHashSystem.ParticleCount;

            if (particleCount == 0)
                return;

            EnsureCapacity(particleCount, config.maxParticles);

            // Chain dependency with spatial hash to safely read grid and positions.
            var handle = JobHandle.CombineDependencies(Dependency, _spatialHashSystem.FinalJobHandle);

            var sourcePositions = _spatialHashSystem.Positions.GetSubArray(0, particleCount);
            var grid = _spatialHashSystem.Grid;
            var workingPositions = _workingPositions.GetSubArray(0, particleCount);
            var densitySlice = _densities.GetSubArray(0, particleCount);
            var lambdaSlice = _lambdas.GetSubArray(0, particleCount);
            var correctionSlice = _corrections.GetSubArray(0, particleCount);

            // Copy predicted positions into the working array (solver modifies in-place).
            handle = new CopyPositionsJob
            {
                Source = sourcePositions,
                Destination = workingPositions
            }.Schedule(particleCount, 128, handle);

            // Precompute artificial pressure reference: W_poly6(Δq · h)
            var inverseReferencePoly6 = 0f;
            if (config.artificialPressureStrength > 0f)
            {
                var deltaQ = config.artificialPressureRadius * config.smoothingRadius;
                var deltaQSq = deltaQ * deltaQ;
                var hSq = config.smoothingRadiusSq;
                var term = hSq - deltaQSq;
                var wDeltaQ = config.poly6Coefficient * term * term * term;
                inverseReferencePoly6 = math.select(0f, math.rcp(wDeltaQ), wDeltaQ > 1e-12f);
            }

            // --- Solver iteration loop ---
            var iterations = math.max(1, config.solverIterations);
            var perIterationStiffness = math.saturate(config.stiffness) / iterations;
            var maxCorrection = config.maxCorrectionFraction * config.smoothingRadius;

            // Precompute clamping bounds for in-solver boundary enforcement.
            var solverBoundsEnabled = bounds.BoundsEnabled;
            var solverBoundsMin = float2.zero;
            var solverBoundsMax = float2.zero;
            if (solverBoundsEnabled != 0)
            {
                var margin = BoundsUtility.EffectiveMargin(bounds.Min, bounds.Max, bounds.Margin);
                solverBoundsMin = bounds.Min + margin;
                solverBoundsMax = bounds.Max - margin;
            }

            for (var iter = 0; iter < iterations; iter++)
            {
                // 1. Compute density from working positions.
                handle = new ComputeDensityJob
                {
                    Positions = workingPositions,
                    Grid = grid,
                    CellSizeInverse = config.cellSizeInv,
                    SmoothingRadiusSq = config.smoothingRadiusSq,
                    Poly6Coefficient = config.poly6Coefficient,
                    ParticleMass = config.uniformParticleMass,
                    Densities = densitySlice
                }.Schedule(particleCount, 64, handle);

                // 2. Compute λ from density and constraint gradients.
                handle = new ComputeLambdaJob
                {
                    Positions = workingPositions,
                    Densities = densitySlice,
                    Grid = grid,
                    CellSizeInverse = config.cellSizeInv,
                    SmoothingRadius = config.smoothingRadius,
                    SmoothingRadiusSq = config.smoothingRadiusSq,
                    SpikyGradCoefficient = config.spikyGradCoefficient,
                    RestDensity = config.restDensity,
                    ParticleMass = config.uniformParticleMass,
                    Epsilon = config.pbfEpsilon,
                    MaxConstraint = config.maxConstraint,
                    Lambdas = lambdaSlice
                }.Schedule(particleCount, 64, handle);

                // 3. Compute position corrections Δp.
                handle = new ComputePositionCorrectionJob
                {
                    Positions = workingPositions,
                    Lambdas = lambdaSlice,
                    Grid = grid,
                    CellSizeInverse = config.cellSizeInv,
                    SmoothingRadius = config.smoothingRadius,
                    SmoothingRadiusSq = config.smoothingRadiusSq,
                    SpikyGradCoefficient = config.spikyGradCoefficient,
                    Poly6Coefficient = config.poly6Coefficient,
                    RestDensity = config.restDensity,
                    ArtificialPressureStrength = config.artificialPressureStrength,
                    ArtificialPressureExponent = config.artificialPressureExponent,
                    InverseReferencePoly6 = inverseReferencePoly6,
                    Corrections = correctionSlice
                }.Schedule(particleCount, 64, handle);

                // 4. Apply Δp with stiffness scaling, magnitude clamping, and boundary enforcement.
                handle = new ApplyPositionCorrectionJob
                {
                    Positions = workingPositions,
                    Corrections = correctionSlice,
                    Stiffness = perIterationStiffness,
                    MaxCorrection = maxCorrection,
                    BoundsEnabled = solverBoundsEnabled,
                    BoundsMin = solverBoundsMin,
                    BoundsMax = solverBoundsMax
                }.Schedule(particleCount, 128, handle);
            }

            // --- Write results back to ECS ---
            handle = new WriteBackSolverResultsJob
            {
                Positions = workingPositions,
                Densities = densitySlice,
                Lambdas = lambdaSlice
            }.ScheduleParallel(_particleQuery, handle);

            Dependency = handle;
        }

        private void EnsureCapacity(int particleCount, int maxParticles)
        {
            var requiredCapacity = math.max(particleCount, maxParticles);

            if (_isAllocated && _workingPositions.Length >= requiredCapacity)
                return;

            if (_isAllocated)
            {
                _workingPositions.Dispose();
                _densities.Dispose();
                _lambdas.Dispose();
                _corrections.Dispose();
            }

            _workingPositions = new NativeArray<float2>(requiredCapacity, Allocator.Persistent);
            _densities = new NativeArray<float>(requiredCapacity, Allocator.Persistent);
            _lambdas = new NativeArray<float>(requiredCapacity, Allocator.Persistent);
            _corrections = new NativeArray<float2>(requiredCapacity, Allocator.Persistent);
            _isAllocated = true;
        }

        protected override void OnDestroy()
        {
            if (!_isAllocated)
                return;

            _workingPositions.Dispose();
            _densities.Dispose();
            _lambdas.Dispose();
            _corrections.Dispose();
        }
    }
}