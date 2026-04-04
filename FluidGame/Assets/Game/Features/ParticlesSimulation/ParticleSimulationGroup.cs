using Unity.Entities;

namespace ParticlesSimulation
{
    /// <summary>
    /// Ordered particle PBF pipeline (2D xy, Burst jobs).
    /// </summary>
    [UpdateInGroup(typeof(SimulationSystemGroup))]
    public partial class ParticleSimulationGroup : ComponentSystemGroup
    {
    }
}
