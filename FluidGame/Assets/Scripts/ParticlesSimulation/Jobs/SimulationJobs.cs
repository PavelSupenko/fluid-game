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
    /// <summary>
    /// Applies external forces (gravity) and writes predictedPosition for the solver.
    /// Only <see cref="ParticlePhase.Fluid"/> particles are integrated;
    /// all other phases keep predictedPosition == position (static until their system moves them).
    /// </summary>
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct PredictPositionsJob : IJobEntity
    {
        public float DeltaTime;
        public float2 Gravity;

        public void Execute(ref ParticleCore core, in ParticleState state)
        {
            if (state.phase != ParticlePhase.Fluid)
            {
                core.predictedPosition = core.position;
                return;
            }

            core.velocity += Gravity * DeltaTime;
            core.predictedPosition = core.position + core.velocity * DeltaTime;
        }
    }

    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct FinalizePositionsJob : IJobEntity
    {
        public float InverseDeltaTime;
        public float MaxSpeedSq;
        public float MaxSpeed;
        public SimulationWorldBounds WorldBounds;

        public void Execute(ref ParticleCore core)
        {
            var predicted = core.predictedPosition;
            var position = core.position;

            var atMinX = false;
            var atMaxX = false;
            var atMinY = false;
            var atMaxY = false;

            if (WorldBounds.BoundsEnabled != 0)
            {
                var margin = BoundsUtility.EffectiveMargin(WorldBounds.Min, WorldBounds.Max, WorldBounds.Margin);
                var clampMin = WorldBounds.Min + margin;
                var clampMax = WorldBounds.Max - margin;

                atMinX = predicted.x <= clampMin.x;
                atMaxX = predicted.x >= clampMax.x;
                atMinY = predicted.y <= clampMin.y;
                atMaxY = predicted.y >= clampMax.y;

                predicted = math.clamp(predicted, clampMin, clampMax);
            }

            var velocity = (predicted - position) * InverseDeltaTime;

            // Inelastic boundary: kill velocity component directed into the wall.
            // For a thick viscous fluid, energy should be fully absorbed on contact.
            if (atMinX && velocity.x < 0f) velocity.x = 0f;
            if (atMaxX && velocity.x > 0f) velocity.x = 0f;
            if (atMinY && velocity.y < 0f) velocity.y = 0f;
            if (atMaxY && velocity.y > 0f) velocity.y = 0f;

            // Hard cap on velocity magnitude to prevent energy runaway.
            var speedSq = math.lengthsq(velocity);
            if (speedSq > MaxSpeedSq)
                velocity *= MaxSpeed * math.rsqrt(speedSq);

            core.velocity = velocity;
            core.position = predicted;
        }
    }
    
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct ApplyScalarFluidDampingJob : IJobEntity
    {
        public float Damping;

        public void Execute(ref ParticleCore core, in ParticleState state)
        {
            var fluidMask = math.select(0f, 1f, state.phase == ParticlePhase.Fluid);
            var factor = math.lerp(1f, 1f - Damping, fluidMask);
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
        public float RestDensity;
        public float QuadScale;
        /// <summary>Maximum random offset per axis to break grid symmetry (e.g., 0.01).</summary>
        public float PositionJitter;
        public uint RandomSeed;

        public void Execute(int index)
        {
            var e = Entities[index];
            var p = Buffer[index];
            // NOTE: CenterOfMass is kept for future rigid shape-matching (rest pose = p.position - CenterOfMass).
            var colorId = (byte)math.min(p.colorIndex, 7);

            // Small deterministic jitter to break grid-aligned symmetry.
            // Without this, particles in a regular grid preserve column structure indefinitely.
            var position = p.position;
            if (PositionJitter > 0f)
            {
                var rng = Unity.Mathematics.Random.CreateFromIndex(RandomSeed + (uint)index);
                position += rng.NextFloat2(-PositionJitter, PositionJitter);
            }

            CommandBuffer.SetComponent(index, e, new ParticleCore
            {
                position = position,
                predictedPosition = position,
                velocity = float2.zero
            });

            CommandBuffer.SetComponent(index, e, new ParticleFluid
            {
                density = 0f,
                restDensity = RestDensity,
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
            var linearColor = new float4(linearRGB, color.w);

            CommandBuffer.SetComponent(index, e, new URPMaterialPropertyBaseColor
            {
                Value = linearColor
            });

            CommandBuffer.SetComponent(index, e, new ParticleOriginalColor
            {
                Value = linearColor
            });

            CommandBuffer.SetComponent(index, e, LocalTransform.FromPositionRotationScale(
                new float3(position.x, position.y, 0f),
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