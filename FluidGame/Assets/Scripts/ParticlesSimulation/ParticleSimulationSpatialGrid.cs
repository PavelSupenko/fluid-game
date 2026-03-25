using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using UnityEngine;

namespace ParticlesSimulation
{
    /// <summary>
    /// Shared simulation buffers: spatial hash (cell → entity) and a per-query-index scratch for PBF delta (avoids ComponentLookup/ref aliasing).
    /// </summary>
    public static class ParticleSimulationSpatialGrid
    {
        private static NativeParallelMultiHashMap<int, Entity> _sCells;
        private static NativeArray<float2> _sPbfDeltaScratch;

        public static NativeParallelMultiHashMap<int, Entity> Cells => _sCells;

        public static NativeArray<float2> PbfDeltaScratch => _sPbfDeltaScratch;

        public static bool IsCreated => _sCells.IsCreated;

        public static void EnsureCapacity(int maxParticles)
        {
            var cap = math.max(256, maxParticles);
            if (!_sCells.IsCreated)
                _sCells = new NativeParallelMultiHashMap<int, Entity>(cap * 2, Allocator.Persistent);

            if (!_sPbfDeltaScratch.IsCreated || _sPbfDeltaScratch.Length < cap)
            {
                if (_sPbfDeltaScratch.IsCreated)
                    _sPbfDeltaScratch.Dispose();
                _sPbfDeltaScratch = new NativeArray<float2>(cap, Allocator.Persistent);
            }
        }

        public static void DisposeAll()
        {
            if (_sCells.IsCreated)
                _sCells.Dispose();
            if (_sPbfDeltaScratch.IsCreated)
                _sPbfDeltaScratch.Dispose();
            _sCells = default;
            _sPbfDeltaScratch = default;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void ResetStatics()
        {
            _sCells = default;
            _sPbfDeltaScratch = default;
        }
    }
}
