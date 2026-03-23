using UnityEngine;

/// <summary>
/// Finds whichever fluid simulation is active (GPU or CPU/Jobs)
/// and provides a unified interface for controllers and renderers.
/// </summary>
public static class FluidSimBridge
{
    public struct SimRef
    {
        public FluidParticle[] Particles;
        public int ParticleCount;
        public ComputeBuffer ParticleBuffer;
        public FluidTypeDefinition[] FluidTypes;
        public Vector2 ContainerMin;
        public Vector2 ContainerMax;
        public bool IsValid;
    }

    /// <summary>
    /// Finds the active simulation in the scene and returns a unified reference.
    /// Call once in Start() and cache the result.
    /// </summary>
    public static SimRef Find()
    {
        // Prefer Jobs sim if both exist and are enabled
        var jobs = Object.FindObjectOfType<FluidSimulationJobs>();
        if (jobs != null && jobs.enabled)
        {
            return new SimRef
            {
                Particles = jobs.Particles,
                ParticleCount = jobs.ParticleCount,
                ParticleBuffer = jobs.ParticleBuffer,
                FluidTypes = jobs.fluidTypes,
                ContainerMin = jobs.containerMin,
                ContainerMax = jobs.containerMax,
                IsValid = true
            };
        }

        var gpu = Object.FindObjectOfType<FluidSimulationGPU>();
        if (gpu != null && gpu.enabled)
        {
            return new SimRef
            {
                Particles = gpu.Particles,
                ParticleCount = gpu.ParticleCount,
                ParticleBuffer = gpu.ParticleBuffer,
                FluidTypes = gpu.fluidTypes,
                ContainerMin = gpu.containerMin,
                ContainerMax = gpu.containerMax,
                IsValid = true
            };
        }

        return new SimRef { IsValid = false };
    }
}
