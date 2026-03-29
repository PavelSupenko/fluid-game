using ParticlesSimulation.Components;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Systems
{
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal struct RigidComChunkJob : IJobChunk
    {
        [ReadOnly] public ComponentTypeHandle<ParticleCore> CoreHandleRo;
        [ReadOnly] public ComponentTypeHandle<ParticleState> StateHandleRo;

        public NativeArray<float2> ChunkSums;
        public NativeArray<int> ChunkCounts;

        public void Execute(in ArchetypeChunk chunk, int chunkIndexInQuery, bool useEnabledMask, in v128 chunkEnabledMask)
        {
            var cores = chunk.GetNativeArray(ref CoreHandleRo);
            var states = chunk.GetNativeArray(ref StateHandleRo);
            
            var enumerator = new ChunkEntityEnumerator(useEnabledMask, chunkEnabledMask, chunk.Count);
            float2 s = 0f;
            var c = 0;
            while (enumerator.NextEntityIndex(out var i))
            {
                var rigid = math.select(0f, 1f, states[i].phase == ParticlePhase.Rigid);
                s += cores[i].predictedPosition * rigid;
                c += (int)rigid;
            }
            ChunkSums[chunkIndexInQuery] = s;
            ChunkCounts[chunkIndexInQuery] = c;
        }
    }

    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal struct RigidComCombineJob : IJob
    {
        [ReadOnly] public NativeArray<float2> ChunkSums;
        [ReadOnly] public NativeArray<int> ChunkCounts;
        public NativeReference<float2> SumOut;
        public NativeReference<int> CountOut;

        public void Execute()
        {
            float2 s = 0f;
            var c = 0;
            for (var i = 0; i < ChunkSums.Length; i++)
            {
                s += ChunkSums[i];
                c += ChunkCounts[i];
            }

            SumOut.Value = s;
            CountOut.Value = c;
        }
    }

    /// <summary>
    /// Computes the center of mass of rigid particles from their predicted positions.
    /// </summary>
    [UpdateInGroup(typeof(ParticleSimulationGroup))]
    [UpdateAfter(typeof(MeltingTriggerSystem))]
    public partial struct RigidComSystem : ISystem
    {
        private EntityQuery query;

        public void OnCreate(ref SystemState state)
        {
            query = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleState, ParticleSimTag>()
                .Build();
            state.RequireForUpdate(query);
        }

        public void OnUpdate(ref SystemState state)
        {
            var n = query.CalculateEntityCount();
            if (n == 0)
            {
                SystemAPI.SetSingleton(new RigidComState { center = float2.zero, count = 0 });
                return;
            }

            var chunkCount = query.CalculateChunkCount();
            var chunkSums = new NativeArray<float2>(chunkCount, Allocator.TempJob);
            var chunkCounts = new NativeArray<int>(chunkCount, Allocator.TempJob);

            var coreHandle = SystemAPI.GetComponentTypeHandle<ParticleCore>(true);
            var phaseHandle = SystemAPI.GetComponentTypeHandle<ParticleState>(true);
            coreHandle.Update(ref state);
            phaseHandle.Update(ref state);

            var handle = new RigidComChunkJob
            {
                CoreHandleRo = coreHandle,
                StateHandleRo = phaseHandle,
                ChunkSums = chunkSums,
                ChunkCounts = chunkCounts
            }.ScheduleParallel(query, state.Dependency);

            var sum = new NativeReference<float2>(Allocator.TempJob);
            var countRef = new NativeReference<int>(Allocator.TempJob);

            handle = new RigidComCombineJob
            {
                ChunkSums = chunkSums,
                ChunkCounts = chunkCounts,
                SumOut = sum,
                CountOut = countRef
            }.Schedule(handle);

            handle.Complete();

            var c = countRef.Value;
            var com = c > 0 ? sum.Value / c : float2.zero;
            SystemAPI.SetSingleton(new RigidComState { center = com, count = c });

            sum.Dispose();
            countRef.Dispose();
            chunkSums.Dispose();
            chunkCounts.Dispose();

            state.Dependency = handle;
        }
    }
}
