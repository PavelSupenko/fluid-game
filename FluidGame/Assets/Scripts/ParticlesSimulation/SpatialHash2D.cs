using Unity.Burst;
using Unity.Mathematics;

namespace ParticlesSimulation
{
    [BurstCompile]
    public static class SpatialHash2D
    {
        public static int HashCell(int cx, int cy)
        {
            return cx * 73856093 ^ cy * 19349663;
        }

        public static int HashPosition(float2 pos, float cellInv)
        {
            var cx = (int)math.floor(pos.x * cellInv);
            var cy = (int)math.floor(pos.y * cellInv);
            return HashCell(cx, cy);
        }

        public static int2 CellCoords(float2 pos, float cellInv)
        {
            var cx = (int)math.floor(pos.x * cellInv);
            var cy = (int)math.floor(pos.y * cellInv);
            return new int2(cx, cy);
        }
    }
}
