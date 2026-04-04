using ParticlesSimulation.Components;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Jobs
{
    /// <summary>
    /// Extracts predicted positions from entities into a flat array
    /// for cache-friendly access by subsequent grid and solver jobs.
    /// </summary>
    [BurstCompile]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct ExtractPredictedPositionsJob : IJobEntity
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<float2> Positions;

        public void Execute([EntityIndexInQuery] int index, in ParticleCore core)
        {
            Positions[index] = core.predictedPosition;
        }
    }

    /// <summary>
    /// Inserts each particle into the spatial hash grid by its predicted position.
    /// Uses the hash map's parallel writer for lock-free concurrent insertion.
    /// </summary>
    [BurstCompile]
    public struct BuildSpatialGridJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> Positions;
        public NativeParallelMultiHashMap<int, int>.ParallelWriter GridWriter;
        public float CellSizeInverse;

        public void Execute(int index)
        {
            var hash = SpatialHash.Hash(Positions[index], CellSizeInverse);
            GridWriter.Add(hash, index);
        }
    }

    /// <summary>
    /// Counts how many neighbors each particle has within the smoothing radius.
    /// Iterates the 3×3 cell neighborhood around each particle's cell.
    /// Used by debug visualization and as a reference pattern for PBF neighbor iteration.
    /// </summary>
    [BurstCompile]
    public struct CountNeighborsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> Positions;
        [ReadOnly] public NativeParallelMultiHashMap<int, int> Grid;
        public float CellSizeInverse;
        public float SmoothingRadiusSq;

        [WriteOnly] public NativeArray<int> NeighborCounts;

        public void Execute(int index)
        {
            var position = Positions[index];
            var cell = SpatialHash.CellCoords(position, CellSizeInverse);
            var count = 0;

            for (var dy = -1; dy <= 1; dy++)
            {
                for (var dx = -1; dx <= 1; dx++)
                {
                    var neighborCellHash = SpatialHash.Hash(cell + new int2(dx, dy));

                    if (!Grid.TryGetFirstValue(neighborCellHash, out var neighborIndex, out var iterator))
                        continue;

                    do
                    {
                        if (neighborIndex == index)
                            continue;

                        var distanceSq = math.lengthsq(Positions[neighborIndex] - position);
                        if (distanceSq < SmoothingRadiusSq)
                            count++;
                    } while (Grid.TryGetNextValue(out neighborIndex, ref iterator));
                }
            }

            NeighborCounts[index] = count;
        }
    }
}
