using Unity.Mathematics;

/// <summary>
/// A spring connecting two particles within the same soft body.
/// Used by PBD solver to maintain shape while allowing deformation.
///
/// Springs are created at init between neighboring particles of the same bodyIndex.
/// During simulation, if stretch exceeds breakThreshold × restLength, the spring is broken.
/// </summary>
public struct SoftBodySpring
{
    public int particleA;       // Index into particle array
    public int particleB;       // Index into particle array
    public float restLength;    // Distance at rest (measured at spawn)
    public float breakThreshold; // Multiplier: spring breaks if length > restLength * breakThreshold
    public int alive;           // 1 = active, 0 = broken
}