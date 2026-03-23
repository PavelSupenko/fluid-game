using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Grid-based spatial hash for fast neighbor queries.
/// Divides space into cells of size equal to the smoothing radius,
/// so neighbor search only needs to check the 3x3 surrounding cells.
/// </summary>
public class SpatialHash
{
    private float cellSize;
    private float cellSizeInv; // Cached 1/cellSize to avoid repeated division
    private Dictionary<int, List<int>> cells;

    // Reusable list to avoid allocations during queries
    private List<int> queryResult;

    public SpatialHash(float cellSize, int estimatedParticleCount)
    {
        this.cellSize = cellSize;
        this.cellSizeInv = 1f / cellSize;

        // Pre-allocate with reasonable capacity
        cells = new Dictionary<int, List<int>>(estimatedParticleCount / 4);
        queryResult = new List<int>(64);
    }

    /// <summary>
    /// Clears all cells and re-inserts every particle.
    /// Called once per frame before neighbor queries.
    /// </summary>
    public void Rebuild(FluidParticle[] particles, int count)
    {
        // Clear existing cells but keep allocated lists
        foreach (var cell in cells.Values)
        {
            cell.Clear();
        }

        for (int i = 0; i < count; i++)
        {
            int hash = HashPosition(particles[i].position);

            if (!cells.TryGetValue(hash, out var list))
            {
                list = new List<int>(16);
                cells[hash] = list;
            }

            list.Add(i);
        }
    }

    /// <summary>
    /// Returns indices of all particles within the 3x3 cell neighborhood
    /// around the given position. Caller must still check actual distance.
    /// </summary>
    public List<int> GetNeighborCandidates(Vector2 position)
    {
        queryResult.Clear();

        int cx = Mathf.FloorToInt(position.x * cellSizeInv);
        int cy = Mathf.FloorToInt(position.y * cellSizeInv);

        // Check 3x3 grid of cells around the particle
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                int hash = HashCell(cx + dx, cy + dy);

                if (cells.TryGetValue(hash, out var list))
                {
                    queryResult.AddRange(list);
                }
            }
        }

        return queryResult;
    }

    /// <summary>
    /// Converts a world position to a cell hash.
    /// </summary>
    private int HashPosition(Vector2 pos)
    {
        int cx = Mathf.FloorToInt(pos.x * cellSizeInv);
        int cy = Mathf.FloorToInt(pos.y * cellSizeInv);
        return HashCell(cx, cy);
    }

    /// <summary>
    /// Combines two cell coordinates into a single hash using large primes.
    /// </summary>
    private int HashCell(int x, int y)
    {
        // Large primes for spatial hashing to minimize collisions
        return x * 73856093 ^ y * 19349663;
    }
}
