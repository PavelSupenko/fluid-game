namespace ParticlesSimulation.Systems
{
    /// <summary>
    /// Design mapping: the PBF pipeline requires rebuilding the spatial hash and density each solver iteration.
    /// Those logical systems are implemented as Burst jobs in <see cref="ParticlePbfJobs"/> and scheduled from
    /// <see cref="ParticlePbfLoopSystem"/> (ClearSpatialHashJob, BuildSpatialHashJob, ComputeDensityJob,
    /// ComputeLambdaJob, ApplyPbfDeltaJob, ApplyRigidShapeJob) so iteration coupling stays deterministic.
    /// </summary>
    internal static class ParticleSimulationLogicalSystems
    {
    }
}
