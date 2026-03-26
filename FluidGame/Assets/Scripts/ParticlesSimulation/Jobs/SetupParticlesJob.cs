using ParticlesSimulation.Components;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Rendering;
using Unity.Transforms;

namespace ParticlesSimulation.Jobs
{
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
                initialLocalPosition = local,
                colorId = colorId
            });

            CommandBuffer.SetComponent(index, e, default(GridHash));
            CommandBuffer.SetComponent(index, e, new ParticleDrawColor { value = p.color });

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