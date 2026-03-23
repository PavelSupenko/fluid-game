using UnityEngine;

/// <summary>
/// Represents a single fluid particle in the simulation.
/// Layout matches the GPU struct in FluidCompute.compute exactly (48 bytes).
/// Field order matters — do not rearrange without updating the compute shader.
/// </summary>
public struct FluidParticle
{
    public Vector2 position;    // 8 bytes  (offset 0)
    public Vector2 velocity;    // 8 bytes  (offset 8)
    public int typeIndex;       // 4 bytes  (offset 16)
    public float density;       // 4 bytes  (offset 20)
    public float pressure;      // 4 bytes  (offset 24)
    public float pad;           // 4 bytes  (offset 28) — alignment padding
    public Color color;         // 16 bytes (offset 32)
    // Total: 48 bytes
}
