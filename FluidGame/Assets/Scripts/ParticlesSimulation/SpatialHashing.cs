using Unity.Mathematics;

namespace ParticlesSimulation
{
    /// <summary>
    /// Burst-compatible utilities for 2D spatial hashing.
    /// Cell size should match the smoothing radius so that neighbor search
    /// only requires iterating the 3×3 neighborhood of cells.
    /// </summary>
    public static class SpatialHash
    {
        // Large primes for spatial hash mixing (widely used in SPH/PBF literature).
        private const int PrimeX = 73856093;
        private const int PrimeY = 19349663;

        /// <summary>
        /// Returns integer cell coordinates for a world-space position.
        /// </summary>
        public static int2 CellCoords(float2 position, float cellSizeInverse)
        {
            return new int2(
                (int)math.floor(position.x * cellSizeInverse),
                (int)math.floor(position.y * cellSizeInverse));
        }

        /// <summary>
        /// Computes a hash key from integer cell coordinates.
        /// </summary>
        public static int Hash(int2 cell)
        {
            return cell.x * PrimeX ^ cell.y * PrimeY;
        }

        /// <summary>
        /// Convenience: hash key for the cell that contains the given position.
        /// </summary>
        public static int Hash(float2 position, float cellSizeInverse)
        {
            return Hash(CellCoords(position, cellSizeInverse));
        }
    }
}
