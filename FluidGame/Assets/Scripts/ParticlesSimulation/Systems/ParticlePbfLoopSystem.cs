using ParticlesSimulation.Components;
using ParticlesSimulation.Jobs;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Systems
{
    /// <summary>
    /// Coupled PBF sub-step: spatial hash rebuild, density from Poly6 neighbors, lambda/delta PBF solve,
    /// then rigid shape matching. Runs multiple solver iterations per frame (each iteration rebuilds the grid
    /// on updated predicted positions). This system corresponds to the logical sequence
    /// SpatialHash → DensityCalculation → ConstraintResolution from the design doc.
    /// </summary>
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(RigidComSystem))]
    public partial struct ParticlePbfLoopSystem : ISystem
    {
        private EntityQuery query;
        private NativeParallelMultiHashMap<int, Entity> _spatialCells;
        private NativeArray<float2> _pbfDeltaScratch;

        public void OnCreate(ref SystemState state)
        {
            query = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleFluid, ParticleState, GridHash, ParticleSimTag>()
                .Build();
            state.RequireForUpdate(query);
            state.RequireForUpdate<SimulationConfig>();
            state.RequireForUpdate<RigidComState>();
        }

        public void OnDestroy(ref SystemState state)
        {
            if (_spatialCells.IsCreated)
                _spatialCells.Dispose();
            if (_pbfDeltaScratch.IsCreated)
                _pbfDeltaScratch.Dispose();
        }

        private void EnsureGridBuffers(int maxParticles)
        {
            var cap = math.max(256, maxParticles);
            if (!_spatialCells.IsCreated)
                _spatialCells = new NativeParallelMultiHashMap<int, Entity>(cap * 2, Allocator.Persistent);

            if (!_pbfDeltaScratch.IsCreated || _pbfDeltaScratch.Length < cap)
            {
                if (_pbfDeltaScratch.IsCreated)
                    _pbfDeltaScratch.Dispose();
                _pbfDeltaScratch = new NativeArray<float2>(cap, Allocator.Persistent);
            }
        }

        public void OnUpdate(ref SystemState state)
        {
            var cfg = SystemAPI.GetSingleton<SimulationConfig>();
            EnsureGridBuffers(cfg.maxParticles);

            var grid = _spatialCells;
            var deltaScratch = _pbfDeltaScratch;
            var com = SystemAPI.GetSingleton<RigidComState>();

            var coresRo = SystemAPI.GetComponentLookup<ParticleCore>(true);
            var fluidsRo = SystemAPI.GetComponentLookup<ParticleFluid>(true);
            var statesRo = SystemAPI.GetComponentLookup<ParticleState>(true);
            coresRo.Update(ref state);
            fluidsRo.Update(ref state);
            statesRo.Update(ref state);

            var handle = state.Dependency;

            handle = new ClearSpatialHashJob { map = grid }.Schedule(handle);
            handle = new BuildSpatialHashJob
            {
                writer = grid.AsParallelWriter(),
                cellInv = cfg.cellSizeInv,
                usePredictedPositions = true
            }.ScheduleParallel(query, handle);

            handle = new ComputeDensityJob
            {
                grid = grid,
                cores = coresRo,
                cellInv = cfg.cellSizeInv,
                h2 = cfg.smoothingRadiusSq,
                poly6Coefficient = cfg.poly6Coefficient,
                uniformParticleMass = cfg.uniformParticleMass
            }.ScheduleParallel(query, handle);

            var iterations = math.max(1, cfg.solverIterations);
            for (var iter = 0; iter < iterations; iter++)
            {
                if (iter > 0)
                {
                    handle = new ClearSpatialHashJob { map = grid }.Schedule(handle);
                    handle = new BuildSpatialHashJob
                    {
                        writer = grid.AsParallelWriter(),
                        cellInv = cfg.cellSizeInv,
                        usePredictedPositions = true
                    }.ScheduleParallel(query, handle);

                    handle = new ComputeDensityJob
                    {
                        grid = grid,
                        cores = coresRo,
                        cellInv = cfg.cellSizeInv,
                        h2 = cfg.smoothingRadiusSq,
                        poly6Coefficient = cfg.poly6Coefficient,
                        uniformParticleMass = cfg.uniformParticleMass
                    }.ScheduleParallel(query, handle);
                }

                coresRo.Update(ref state);
                fluidsRo.Update(ref state);
                statesRo.Update(ref state);

                handle = new ComputeLambdaJob
                {
                    grid = grid,
                    cores = coresRo,
                    states = statesRo,
                    cellInv = cfg.cellSizeInv,
                    h = cfg.smoothingRadius,
                    spikyGradCoefficient = cfg.spikyGradCoefficient,
                    epsilon = cfg.pbfEpsilon,
                    stiffness = cfg.stiffness,
                    uniformParticleMass = cfg.uniformParticleMass
                }.ScheduleParallel(query, handle);

                coresRo.Update(ref state);
                fluidsRo.Update(ref state);
                statesRo.Update(ref state);

                handle = new AccumulatePbfDeltaJob
                {
                    grid = grid,
                    cores = coresRo,
                    fluids = fluidsRo,
                    states = statesRo,
                    deltaOut = deltaScratch,
                    cellInv = cfg.cellSizeInv,
                    h = cfg.smoothingRadius,
                    spikyGradCoefficient = cfg.spikyGradCoefficient,
                    deltaScale = cfg.deltaScale,
                    uniformParticleMass = cfg.uniformParticleMass
                }.ScheduleParallel(query, handle);

                handle = new ApplyPbfDeltaFromBufferJob
                {
                    deltaIn = deltaScratch
                }.ScheduleParallel(query, handle);

                handle = new ApplyRigidShapeJob
                {
                    rigidCenter = com.center,
                    rigidCount = com.count,
                    shapeStiffness = cfg.rigidShapeStiffness
                }.ScheduleParallel(query, handle);
            }

            state.Dependency = handle;
        }
    }
}
