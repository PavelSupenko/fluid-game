using ParticlesSimulation.Components;
using ParticlesSimulation.Jobs;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Systems
{
    /// <summary>
    /// Builds the spatial hash grid from predicted positions each frame.
    /// Exposes the grid, position array, and neighbor counts for dependent systems
    /// (PBF solver, debug visualization).
    /// Pipeline: Clock → Prediction → <b>SpatialHash</b> → [PBF solver] → Finalization.
    /// </summary>
    /// <remarks>
    /// Uses <see cref="SystemBase"/> (class-based) so that dependent systems can cache
    /// a managed reference and read the native containers directly.
    /// </remarks>
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(PredictionSystem))]
    [UpdateBefore(typeof(FinalizationSystem))]
    public partial class SpatialHashGridSystem : SystemBase
    {
        private NativeParallelMultiHashMap<int, int> _grid;
        private NativeArray<float2> _positions;
        private NativeArray<int> _neighborCounts;
        private EntityQuery _particleQuery;
        private bool _isAllocated;

        /// <summary>Read-only view of the spatial grid. Valid after this system updates.</summary>
        public NativeParallelMultiHashMap<int, int> Grid => _grid;

        /// <summary>Predicted positions indexed by entity query order. Valid after this system updates.</summary>
        public NativeArray<float2> Positions => _positions;

        /// <summary>Neighbor counts per particle. Valid after this system updates.</summary>
        public NativeArray<int> NeighborCounts => _neighborCounts;

        /// <summary>Number of active particles this frame.</summary>
        public int ParticleCount { get; private set; }

        /// <summary>
        /// The final job handle scheduled by this system.
        /// Consuming systems must complete or depend on this before reading the native arrays.
        /// </summary>
        public JobHandle FinalJobHandle { get; private set; }

        protected override void OnCreate()
        {
            _particleQuery = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleState, ParticleSimulatedTag>()
                .Build();

            RequireForUpdate(_particleQuery);
            RequireForUpdate<SimulationConfig>();
        }

        protected override void OnUpdate()
        {
            var config = SystemAPI.GetSingleton<SimulationConfig>();
            var particleCount = _particleQuery.CalculateEntityCount();
            ParticleCount = particleCount;

            if (particleCount == 0)
                return;

            EnsureCapacity(particleCount, config.maxParticles);

            // Step 1: extract predicted positions into a flat array for cache-friendly access.
            var positionSlice = _positions.GetSubArray(0, particleCount);

            var extractHandle = new ExtractPredictedPositionsJob
            {
                Positions = positionSlice
            }.ScheduleParallel(_particleQuery, Dependency);

            // Step 2: clear and rebuild the spatial grid.
            _grid.Clear();
            if (_grid.Capacity < particleCount * 2)
                _grid.Capacity = particleCount * 2;

            var buildHandle = new BuildSpatialGridJob
            {
                Positions = positionSlice,
                GridWriter = _grid.AsParallelWriter(),
                CellSizeInverse = config.cellSizeInv
            }.Schedule(particleCount, 128, extractHandle);

            // Step 3: count neighbors per particle (useful for debug and PBF diagnostics).
            var neighborSlice = _neighborCounts.GetSubArray(0, particleCount);

            var countHandle = new CountNeighborsJob
            {
                Positions = positionSlice,
                Grid = _grid,
                CellSizeInverse = config.cellSizeInv,
                SmoothingRadiusSq = config.smoothingRadiusSq,
                NeighborCounts = neighborSlice
            }.Schedule(particleCount, 64, buildHandle);

            Dependency = countHandle;
            FinalJobHandle = countHandle;
        }

        private void EnsureCapacity(int particleCount, int maxParticles)
        {
            var requiredCapacity = math.max(particleCount, maxParticles);

            if (_isAllocated && _positions.Length >= requiredCapacity)
                return;

            if (_isAllocated)
            {
                _grid.Dispose();
                _positions.Dispose();
                _neighborCounts.Dispose();
            }

            _grid = new NativeParallelMultiHashMap<int, int>(requiredCapacity * 2, Allocator.Persistent);
            _positions = new NativeArray<float2>(requiredCapacity, Allocator.Persistent);
            _neighborCounts = new NativeArray<int>(requiredCapacity, Allocator.Persistent);
            _isAllocated = true;
        }

        protected override void OnDestroy()
        {
            if (!_isAllocated)
                return;

            _grid.Dispose();
            _positions.Dispose();
            _neighborCounts.Dispose();
        }
    }
}
