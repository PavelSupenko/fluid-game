using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation
{
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal struct RigidComAccumulateJob : IJob
    {
        [ReadOnly] public NativeArray<ParticleCore> cores;
        [ReadOnly] public NativeArray<ParticleState> states;
        public NativeReference<float2> sum;
        public NativeReference<int> count;

        public void Execute()
        {
            float2 s = 0f;
            var c = 0;
            for (var i = 0; i < cores.Length; i++)
            {
                var rigid = math.select(0f, 1f, states[i].phase == ParticlePhase.Rigid);
                s += cores[i].predictedPosition * rigid;
                c += (int)rigid;
            }

            sum.Value = s;
            count.Value = c;
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
            state.RequireForUpdate<SimulationConfig>();
        }

        public void OnUpdate(ref SystemState state)
        {
            var n = query.CalculateEntityCount();
            if (n == 0)
            {
                SystemAPI.SetSingleton(new RigidComState { center = float2.zero, count = 0 });
                return;
            }

            var cores = query.ToComponentDataArray<ParticleCore>(Allocator.TempJob);
            var states = query.ToComponentDataArray<ParticleState>(Allocator.TempJob);
            var sum = new NativeReference<float2>(Allocator.TempJob);
            var count = new NativeReference<int>(Allocator.TempJob);

            var handle = new RigidComAccumulateJob
            {
                cores = cores,
                states = states,
                sum = sum,
                count = count
            }.Schedule(state.Dependency);

            handle.Complete();

            var c = count.Value;
            var com = c > 0 ? sum.Value / c : float2.zero;
            SystemAPI.SetSingleton(new RigidComState { center = com, count = c });

            sum.Dispose();
            count.Dispose();
            cores.Dispose();
            states.Dispose();
        }
    }
}
