using UnityEngine;

/// <summary>
/// Represents a single fluid particle in the simulation.
/// Stored as a struct for cache-friendly iteration.
/// </summary>
public struct FluidParticle
{
    public Vector2 position;
    public Vector2 velocity;
    public int typeIndex;   // Index into FluidSimulation.fluidTypes array
    public Color color;
}
