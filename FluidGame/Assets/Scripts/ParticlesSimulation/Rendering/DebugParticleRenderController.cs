using ParticlesSimulation.Components;
using ParticlesSimulation.Systems;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Rendering;
using Unity.Transforms;
using UnityEngine;

namespace ParticlesSimulation.Debug
{
    // ─────────────────────────────────────────────────────────────────────────
    //  Data types
    // ─────────────────────────────────────────────────────────────────────────

    public enum DebugParticleMode : byte
    {
        /// <summary>Original image/spawn colors — no debug override.</summary>
        Normal = 0,

        /// <summary>Rigid = blue, Fluid = orange.</summary>
        Phase = 1,

        /// <summary>Heatmap by velocity magnitude.</summary>
        Speed = 2,

        /// <summary>Heatmap by spatial-hash neighbor count.</summary>
        NeighborCount = 3,

        /// <summary>Heatmap by PBF density (placeholder until solver is online).</summary>
        Density = 4,
    }

    /// <summary>
    /// Singleton ECS component driven by <see cref="DebugParticleRenderController"/>.
    /// The <see cref="PreviousMode"/> field is managed by the ECS system to detect transitions.
    /// </summary>
    public struct DebugRenderState : IComponentData
    {
        public DebugParticleMode ActiveMode;
        public DebugParticleMode PreviousMode;
        public float MaxSpeedDisplay;
        public int MaxNeighborDisplay;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Utility
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Burst-compatible five-stop heatmap: Blue → Cyan → Green → Yellow → Red.
    /// </summary>
    public static class ColorGradient
    {
        public static float4 Heatmap(float normalizedValue)
        {
            var t = math.saturate(normalizedValue);
            float3 color;

            if (t < 0.25f)
                color = math.lerp(new float3(0f, 0f, 1f), new float3(0f, 1f, 1f), t * 4f);
            else if (t < 0.5f)
                color = math.lerp(new float3(0f, 1f, 1f), new float3(0f, 1f, 0f), (t - 0.25f) * 4f);
            else if (t < 0.75f)
                color = math.lerp(new float3(0f, 1f, 0f), new float3(1f, 1f, 0f), (t - 0.5f) * 4f);
            else
                color = math.lerp(new float3(1f, 1f, 0f), new float3(1f, 0f, 0f), (t - 0.75f) * 4f);

            return new float4(color, 1f);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  MonoBehaviour controller (inspector interface)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Inspector-driven controller for debug particle visualization.
    /// Creates a <see cref="DebugRenderState"/> singleton and pushes settings each frame.
    /// If this component is absent from the scene, no debug overhead is incurred.
    /// </summary>
    [DefaultExecutionOrder(-400)]
    public sealed class DebugParticleRenderController : MonoBehaviour
    {
        [Header("Debug Visualization")]
        [Tooltip("Choose which property to visualize on particles.")]
        [SerializeField]
        private DebugParticleMode _mode = DebugParticleMode.Normal;

        [Tooltip("Velocity magnitude mapped to 1.0 in the Speed heatmap.")]
        [SerializeField]
        private float _maxSpeedDisplay = 10f;

        [Tooltip("Neighbor count mapped to 1.0 in the NeighborCount heatmap.")]
        [SerializeField]
        private int _maxNeighborDisplay = 30;

        private Entity _debugEntity;
        private bool _entityCreated;

        private void Awake()
        {
            if (!Application.isPlaying)
                return;

            var world = World.DefaultGameObjectInjectionWorld;
            if (world == null || !world.IsCreated)
                return;

            var entityManager = world.EntityManager;
            _debugEntity = entityManager.CreateEntity();
            entityManager.AddComponentData(_debugEntity, new DebugRenderState
            {
                ActiveMode = _mode,
                PreviousMode = _mode,
                MaxSpeedDisplay = _maxSpeedDisplay,
                MaxNeighborDisplay = _maxNeighborDisplay
            });
            _entityCreated = true;
        }

        private void Update()
        {
            if (!_entityCreated)
                return;

            var world = World.DefaultGameObjectInjectionWorld;
            if (world == null || !world.IsCreated)
                return;

            var entityManager = world.EntityManager;
            if (!entityManager.Exists(_debugEntity))
                return;

            // Preserve PreviousMode (managed by the ECS system), update everything else.
            var current = entityManager.GetComponentData<DebugRenderState>(_debugEntity);
            current.ActiveMode = _mode;
            current.MaxSpeedDisplay = _maxSpeedDisplay;
            current.MaxNeighborDisplay = _maxNeighborDisplay;
            entityManager.SetComponentData(_debugEntity, current);
        }

        private void OnDestroy()
        {
            if (!_entityCreated)
                return;

            var world = World.DefaultGameObjectInjectionWorld;
            if (world != null && world.IsCreated && world.EntityManager.Exists(_debugEntity))
                world.EntityManager.DestroyEntity(_debugEntity);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Debug color jobs
    // ─────────────────────────────────────────────────────────────────────────

    [BurstCompile]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct RestoreOriginalColorsJob : IJobEntity
    {
        public void Execute(ref URPMaterialPropertyBaseColor baseColor, in ParticleOriginalColor originalColor)
        {
            baseColor.Value = originalColor.Value;
        }
    }

    [BurstCompile]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct ApplyPhaseColorsJob : IJobEntity
    {
        public void Execute(ref URPMaterialPropertyBaseColor baseColor, in ParticleState state)
        {
            baseColor.Value = state.phase == ParticlePhase.Fluid
                ? new float4(1f, 0.5f, 0.1f, 1f) // Orange
                : new float4(0.2f, 0.5f, 1f, 1f); // Blue
        }
    }

    [BurstCompile]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct ApplySpeedColorsJob : IJobEntity
    {
        public float InverseMaxSpeed;

        public void Execute(ref URPMaterialPropertyBaseColor baseColor, in ParticleCore core)
        {
            var speed = math.length(core.velocity);
            baseColor.Value = ColorGradient.Heatmap(speed * InverseMaxSpeed);
        }
    }

    [BurstCompile]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct ApplyNeighborCountColorsJob : IJobEntity
    {
        [ReadOnly] public NativeArray<int> NeighborCounts;
        public float InverseMaxNeighbors;

        public void Execute([EntityIndexInQuery] int index, ref URPMaterialPropertyBaseColor baseColor)
        {
            var count = NeighborCounts[index];
            baseColor.Value = ColorGradient.Heatmap(count * InverseMaxNeighbors);
        }
    }

    [BurstCompile]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct ApplyDensityColorsJob : IJobEntity
    {
        public void Execute(ref URPMaterialPropertyBaseColor baseColor, in ParticleFluid fluid)
        {
            // Density visualization: ratio of current density to rest density.
            // Will produce a meaningful gradient once the PBF solver is online.
            var ratio = math.select(0f, fluid.density / fluid.restDensity, fluid.restDensity > 0f);
            baseColor.Value = ColorGradient.Heatmap(math.saturate(ratio));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  ECS system
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Overrides <see cref="URPMaterialPropertyBaseColor"/> based on the active
    /// <see cref="DebugParticleMode"/>. Does nothing in <see cref="DebugParticleMode.Normal"/>
    /// unless transitioning back from a debug mode (one-shot color restore).
    /// </summary>
    [UpdateInGroup(typeof(SimulationSystemGroup))]
    [UpdateAfter(typeof(ParticleSimulationGroup))]
    [UpdateBefore(typeof(TransformSystemGroup))]
    public partial class DebugParticleColorSystem : SystemBase
    {
        private SpatialHashGridSystem _spatialHashSystem;
        private EntityQuery _particleQuery;

        protected override void OnCreate()
        {
            _spatialHashSystem = World.GetExistingSystemManaged<SpatialHashGridSystem>();

            _particleQuery = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleFluid, ParticleState, ParticleSimulatedTag,
                         URPMaterialPropertyBaseColor, ParticleOriginalColor>()
                .Build();

            RequireForUpdate<DebugRenderState>();
            RequireForUpdate(_particleQuery);
        }

        protected override void OnUpdate()
        {
            var debugState = SystemAPI.GetSingletonRW<DebugRenderState>();
            var activeMode = debugState.ValueRO.ActiveMode;
            var previousMode = debugState.ValueRO.PreviousMode;

            // Nothing to do when we're already in Normal and were in Normal last frame.
            if (activeMode == DebugParticleMode.Normal && previousMode == DebugParticleMode.Normal)
                return;

            debugState.ValueRW.PreviousMode = activeMode;

            switch (activeMode)
            {
                case DebugParticleMode.Normal:
                    Dependency = new RestoreOriginalColorsJob()
                        .ScheduleParallel(_particleQuery, Dependency);
                    break;

                case DebugParticleMode.Phase:
                    Dependency = new ApplyPhaseColorsJob()
                        .ScheduleParallel(_particleQuery, Dependency);
                    break;

                case DebugParticleMode.Speed:
                    var maxSpeed = math.max(debugState.ValueRO.MaxSpeedDisplay, 0.001f);
                    Dependency = new ApplySpeedColorsJob
                    {
                        InverseMaxSpeed = 1f / maxSpeed
                    }.ScheduleParallel(_particleQuery, Dependency);
                    break;

                case DebugParticleMode.NeighborCount:
                    if (_spatialHashSystem == null)
                        break;
                    // Ensure spatial hash jobs are complete before reading their native arrays.
                    _spatialHashSystem.FinalJobHandle.Complete();
                    var particleCount = _spatialHashSystem.ParticleCount;
                    if (particleCount == 0)
                        break;
                    var maxNeighbors = math.max(debugState.ValueRO.MaxNeighborDisplay, 1);
                    Dependency = new ApplyNeighborCountColorsJob
                    {
                        NeighborCounts = _spatialHashSystem.NeighborCounts.GetSubArray(0, particleCount),
                        InverseMaxNeighbors = 1f / (float)maxNeighbors
                    }.ScheduleParallel(_particleQuery, Dependency);
                    break;

                case DebugParticleMode.Density:
                    Dependency = new ApplyDensityColorsJob()
                        .ScheduleParallel(_particleQuery, Dependency);
                    break;
            }
        }
    }
}
