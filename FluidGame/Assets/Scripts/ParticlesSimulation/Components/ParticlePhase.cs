namespace ParticlesSimulation
{
    /// <summary>
    /// Rigid particles preserve image structure via shape matching; fluid particles use PBF.
    /// </summary>
    public enum ParticlePhase : byte
    {
        Rigid = 0,
        Fluid = 1
    }
}
