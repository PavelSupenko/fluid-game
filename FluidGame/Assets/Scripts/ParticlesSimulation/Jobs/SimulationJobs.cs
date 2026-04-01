using ParticlesSimulation.Components;
using Unity.Mathematics;
using Unity.Entities;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Rendering;
using Unity.Transforms;

namespace ParticlesSimulation.Jobs
{
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct IntegratePositionsJob : IJobEntity
    {
        public float invDt;
        public SimulationWorldBounds WorldBounds;

        public void Execute(ref ParticleCore core)
        {
            var pred = core.predictedPosition;
            var pos = core.position;
            if (WorldBounds.BoundsEnabled != 0)
            {
                var ext = WorldBounds.Max - WorldBounds.Min;
                var margin = math.min(WorldBounds.Margin, math.max(0f, 0.49f * math.cmin(ext)));
                var min = WorldBounds.Min + margin;
                var max = WorldBounds.Max - margin;
                pred = math.clamp(pred, min, max);
            }

            core.velocity = (pred - pos) * invDt;
            core.position = pred;
        }
    }
    
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct ApplyScalarFluidDampingJob : IJobEntity
    {
        public float damping;

        public void Execute(ref ParticleCore core, in ParticleState state)
        {
            var fluidMask = math.select(0f, 1f, state.phase == ParticlePhase.Fluid);
            var factor = math.lerp(1f, 1f - damping, fluidMask);
            core.velocity *= factor;
        }
    }
    
        [BurstCompile]
    public struct SetupParticlesJob : IJobParallelFor
    {
        public EntityCommandBuffer.ParallelWriter CommandBuffer;
        [ReadOnly] public NativeArray<Entity> Entities;
        [ReadOnly] public NativeArray<SpawnParticle> Buffer;

        public float2 CenterOfMass;
        public float ParticleMass;
        public float RestDensity;
        public float QuadScale;

        public void Execute(int index)
        {
            var e = Entities[index];
            var p = Buffer[index];
            var local = p.position - CenterOfMass;
            var colorId = (byte)math.min(p.colorIndex, 7);

            CommandBuffer.SetComponent(index, e, new ParticleCore
            {
                position = p.position,
                predictedPosition = p.position,
                velocity = float2.zero,
                mass = ParticleMass
            });

            CommandBuffer.SetComponent(index, e, new ParticleFluid
            {
                density = 0f,
                pressure = 0f,
                restDensity = RestDensity,
                mass = ParticleMass,
                lambda = 0f
            });

            CommandBuffer.SetComponent(index, e, new ParticleState
            {
                phase = ParticlePhase.Fluid,
                colorId = colorId
            });

            // Burst-compatible Gamma to Linear conversion
            var color = p.color;
            var sRGB = color.xyz;
            var linearRGB = math.select(
                math.pow((sRGB + 0.055f) / 1.055f, 2.4f),
                sRGB / 12.92f,
                sRGB <= 0.04045f);

            CommandBuffer.SetComponent(index, e, new URPMaterialPropertyBaseColor
            {
                Value = new float4(linearRGB, color.w)
            });

            CommandBuffer.SetComponent(index, e, LocalTransform.FromPositionRotationScale(
                new float3(p.position.x, p.position.y, 0f),
                quaternion.identity,
                QuadScale));
        }
    }

    public struct SpawnParticle
    {
        public float2 position;
        public float4 color;
        public int colorIndex;
    }
}