using ParticlesSimulation.Components;
using ParticlesSimulation.Jobs;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Rendering;
using Unity.Transforms;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Serialization;

namespace ParticlesSimulation
{
    /// <summary>
    /// Spawns ECS particles from <see cref="ImageToFluid"/> output (Core) or a procedural grid,
    /// wires singleton simulation data, registers Entities Graphics (<see cref="RenderMeshArray"/>,
    /// <see cref="MaterialMeshInfo"/>, <see cref="URPMaterialPropertyBaseColor"/>), and initializes the spatial hash.
    /// </summary>
    [DefaultExecutionOrder(-500)]
    public sealed class ParticleSimulationBootstrap : MonoBehaviour
    {
        [Header("Sources (Core)")]
        [SerializeField]
        private ImageToFluid _imageToFluid;

        [Tooltip("Optional. When set, simulation is clamped to this rect (world XY) and particles spawn inside it.")]
        [SerializeField]
        private ParticleSimulationBounds _simulationBounds;

        [Header("Fallback grid (no image)")]
        [SerializeField]
        private int _fallbackGridX = 32;

        [SerializeField]
        private int _fallbackGridY = 48;

        [SerializeField]
        private float2 _fallbackOrigin = new float2(-1.6f, -2.2f);

        [SerializeField]
        private float _fallbackSpacing = 0.08f;

        [Header("Simulation tuning")]
        [SerializeField]
        private float _smoothingRadius = 0.12f;

        [SerializeField]
        private float _meltLineY = -1.2f;

        [SerializeField]
        private float _gravityY = -12f;

        [Tooltip("Per-frame velocity damping for fluid particles (0 = no damping, 1 = full stop). " +
                 "Higher values = thicker fluid. 0.3 gives honey-like behavior.")]
        [Range(0f, 1f)]
        [SerializeField]
        private float _fluidDamping = 0.3f;

        [SerializeField]
        private float _stiffness = 0.5f;

        [SerializeField]
        private float _rigidShapeStiffness = 0.65f;

        [SerializeField]
        private int _solverIterations = 4;

        [SerializeField]
        private float _particleMass = 1f;

        [Tooltip("Manual rest density. Ignored when Auto Estimate is enabled.")]
        [SerializeField]
        private float _restDensity = 300;

        [Tooltip("Automatically compute rest density from particle spacing and smoothing radius. " +
                 "Recommended for stable simulation without manual tuning.")]
        [SerializeField]
        private bool _autoEstimateRestDensity = true;

        [Header("Entities Graphics (URP)")]
        [Tooltip("URP Lit/Unlit material that uses _BaseColor (Entities Graphics compatible).")]
        [SerializeField]
        private Material _particleMaterial;

        [Tooltip("Optional; if null a centered unit quad (1×1) is created at runtime.")]
        [SerializeField]
        private Mesh _quadMesh;

        [SerializeField]
        private float _quadHalfExtent = 0.035f;

        [SerializeField]
        private int _renderingLayer;

        private EntityManager entityManager;
        private Entity singletonEntity;
        private bool spawned;
        private Mesh _runtimeQuadMesh;

        private void Awake()
        {
            if (!Application.isPlaying)
                return;

            var world = World.DefaultGameObjectInjectionWorld;
            if (world == null || !world.IsCreated)
            {
                UnityEngine.Debug.LogError("[ParticleSimulationBootstrap] Default ECS world not available.");
                return;
            }

            entityManager = world.EntityManager;
            CreateSingletons();
            SpawnParticles();
        }

        private void OnDestroy()
        {
            if (_runtimeQuadMesh != null)
                Destroy(_runtimeQuadMesh);
        }

        private void CreateSingletons()
        {
            singletonEntity = entityManager.CreateEntity();
            entityManager.AddComponentData(singletonEntity, new SpatialGridMapTag());

            var maxEstimate = 1;
            if (_imageToFluid != null)
            {
                _imageToFluid.TryParseImage();
                if (_imageToFluid.IsReady)
                    maxEstimate = math.max(_imageToFluid.GeneratedParticleCount, 1);
                else
                    maxEstimate = _fallbackGridX * _fallbackGridY;
            }
            else
            {
                maxEstimate = _fallbackGridX * _fallbackGridY;
            }

            var cfg = ConfigUtility.CreateDefault(maxEstimate);
            cfg.gravityY = _gravityY;
            cfg.meltLineY = _meltLineY;
            cfg.fluidDamping = _fluidDamping;
            cfg.stiffness = _stiffness;
            cfg.rigidShapeStiffness = _rigidShapeStiffness;
            cfg.solverIterations = _solverIterations;
            ConfigUtility.ApplySmoothingRadius(ref cfg, _smoothingRadius);
            cfg.deltaTime = Time.fixedDeltaTime > 0f ? Time.fixedDeltaTime : Time.deltaTime;
            cfg.maxParticles = math.max(cfg.maxParticles, maxEstimate + 256);
            cfg.uniformParticleMass = _particleMass;
            if (_autoEstimateRestDensity)
            {
                cfg.restDensity = ConfigUtility.EstimateRestDensity(in cfg, _fallbackSpacing);
                UnityEngine.Debug.Log($"[ParticleSimulationBootstrap] Auto-estimated restDensity = {cfg.restDensity:F1} " +
                                      $"(spacing={_fallbackSpacing}, h={_smoothingRadius})");
            }
            else
            {
                cfg.restDensity = _restDensity;
            }

            entityManager.AddComponentData(singletonEntity, cfg);

            var worldBounds = new SimulationWorldBounds { BoundsEnabled = 0 };
            if (_simulationBounds != null &&
                RectTransformSimulationBoundsUtility.TryGetWorldAabbXY(_simulationBounds.AreaRect, out var wMin, out var wMax))
            {
                worldBounds = new SimulationWorldBounds
                {
                    BoundsEnabled = 1,
                    Min = wMin,
                    Max = wMax,
                    Margin = _simulationBounds.Margin
                };
            }

            entityManager.AddComponentData(singletonEntity, worldBounds);
        }

        private void SpawnParticles()
        {
            if (spawned)
                return;

            if (_particleMaterial == null)
            {
                UnityEngine.Debug.LogError(
                    "[ParticleSimulationBootstrap] No particle material assigned - cannot spawn entities with graphics components.");
                return;
            }

            var buffer = new NativeList<SpawnParticle>(Allocator.TempJob);
            try
            {
                if (_imageToFluid != null && _imageToFluid.IsReady)
                {
                    AppendFromImage(buffer);
                    if (TryGetSpawnInnerRect(out var innerMin, out var innerMax))
                        RemapSpawnBufferToInnerRect(buffer, innerMin, innerMax);
                }
                else
                {
                    AppendFallbackGrid(buffer);
                }

                if (buffer.Length == 0)
                {
                    UnityEngine.Debug.LogWarning("[ParticleSimulationBootstrap] No particles spawned.");
                    return;
                }

                var centerOfMass = float2.zero;
                for (var i = 0; i < buffer.Length; i++)
                    centerOfMass += buffer[i].position;
                centerOfMass /= buffer.Length;

                var archetype = entityManager.CreateArchetype(
                    typeof(ParticleCore),
                    typeof(ParticleFluid),
                    typeof(ParticleState),
                    typeof(ParticleSimulatedTag),
                    typeof(ParticleOriginalColor),
                    typeof(URPMaterialPropertyBaseColor),
                    typeof(LocalTransform));

                var prototype = entityManager.CreateEntity(archetype);
                var mesh = _quadMesh;
                if (mesh == null)
                {
                    _runtimeQuadMesh = CreateUnitQuadMesh();
                    mesh = _runtimeQuadMesh;
                }

                var renderMeshArray = new RenderMeshArray(new[] { _particleMaterial }, new[] { mesh });
                var desc = new RenderMeshDescription(
                    shadowCastingMode: ShadowCastingMode.Off,
                    receiveShadows: false,
                    motionVectorGenerationMode: MotionVectorGenerationMode.ForceNoMotion,
                    layer: _renderingLayer);

                RenderMeshUtility.AddComponents(
                    prototype,
                    entityManager,
                    desc,
                    renderMeshArray,
                    MaterialMeshInfo.FromRenderMeshArrayIndices(0, 0));

                using var entities = new NativeArray<Entity>(buffer.Length, Allocator.TempJob);
                entityManager.Instantiate(prototype, entities);
                entityManager.DestroyEntity(prototype);

                using var commandBuffer = new EntityCommandBuffer(Allocator.TempJob);

                // Use the config's restDensity which may have been auto-estimated.
                var spawnConfig = entityManager.GetComponentData<SimulationConfig>(singletonEntity);

                var setupJob = new SetupParticlesJob
                {
                    Entities = entities,
                    Buffer = buffer.AsArray(),
                    CommandBuffer = commandBuffer.AsParallelWriter(),
                    CenterOfMass = centerOfMass,
                    RestDensity = spawnConfig.restDensity,
                    QuadScale = _quadHalfExtent * 2f,
                    PositionJitter = _fallbackSpacing * 0.02f,
                    RandomSeed = 42u
                };

                setupJob.Schedule(buffer.Length, 64).Complete();
                commandBuffer.Playback(entityManager);

                var cfg = entityManager.GetComponentData<SimulationConfig>(singletonEntity);
                cfg.maxParticles = math.max(cfg.maxParticles, buffer.Length + 256);
                entityManager.SetComponentData(singletonEntity, cfg);

                spawned = true;
            }
            finally
            {
                buffer.Dispose();
            }
        }

        private static Mesh CreateUnitQuadMesh()
        {
            var m = new Mesh { name = "ParticleECSUnitQuad" };
            m.vertices = new[]
            {
                new Vector3(-0.5f, -0.5f, 0f),
                new Vector3(-0.5f, 0.5f, 0f),
                new Vector3(0.5f, 0.5f, 0f),
                new Vector3(0.5f, -0.5f, 0f)
            };
            m.uv = new[]
            {
                new Vector2(0f, 0f),
                new Vector2(0f, 1f),
                new Vector2(1f, 1f),
                new Vector2(1f, 0f)
            };
            m.triangles = new[] { 0, 1, 2, 0, 2, 3 };
            m.RecalculateNormals();
            m.RecalculateBounds();
            return m;
        }

        private void AppendFromImage(NativeList<SpawnParticle> buffer)
        {
            var parts = _imageToFluid.GeneratedParticles;
            var types = _imageToFluid.GeneratedFluidTypes;
            var count = _imageToFluid.GeneratedParticleCount;

            for (var i = 0; i < count; i++)
            {
                var fp = parts[i];
                var idx = math.clamp(fp.typeIndex, 0, types.Length - 1);
                var col = types[idx].color;
                buffer.Add(new SpawnParticle
                {
                    position = new float2(fp.position.x, fp.position.y),
                    color = new float4(col.r, col.g, col.b, col.a),
                    colorIndex = idx
                });
            }
        }

        private void AppendFallbackGrid(NativeList<SpawnParticle> buffer)
        {
            if (TryGetSpawnInnerRect(out var innerMin, out var innerMax))
            {
                var size = innerMax - innerMin;
                var nx = math.max(1, _fallbackGridX - 1);
                var ny = math.max(1, _fallbackGridY - 1);
                var spacingX = size.x / nx;
                var spacingY = size.y / ny;
                for (var y = 0; y < _fallbackGridY; y++)
                {
                    for (var x = 0; x < _fallbackGridX; x++)
                    {
                        var pos = innerMin + new float2(x * spacingX, y * spacingY);
                        var t = (float)(x + y) / math.max(1, _fallbackGridX + _fallbackGridY - 2);
                        var col = Color.HSVToRGB(t, 0.65f, 0.95f);
                        buffer.Add(new SpawnParticle
                        {
                            position = pos,
                            color = new float4(col.r, col.g, col.b, 1f),
                            colorIndex = (int)(t * 7f) & 7
                        });
                    }
                }

                return;
            }

            for (var y = 0; y < _fallbackGridY; y++)
            {
                for (var x = 0; x < _fallbackGridX; x++)
                {
                    var pos = _fallbackOrigin + new float2(x * _fallbackSpacing, y * _fallbackSpacing);
                    var t = (float)(x + y) / math.max(1, _fallbackGridX + _fallbackGridY - 2);
                    var col = Color.HSVToRGB(t, 0.65f, 0.95f);
                    buffer.Add(new SpawnParticle
                    {
                        position = pos,
                        color = new float4(col.r, col.g, col.b, 1f),
                        colorIndex = (int)(t * 7f) & 7
                    });
                }
            }
        }

        private bool TryGetSpawnInnerRect(out float2 innerMin, out float2 innerMax)
        {
            innerMin = innerMax = default;
            if (_simulationBounds == null || _simulationBounds.AreaRect == null)
                return false;
            if (!RectTransformSimulationBoundsUtility.TryGetWorldAabbXY(_simulationBounds.AreaRect, out var wMin, out var wMax))
                return false;

            ComputeInnerClampBox(wMin, wMax, _simulationBounds.Margin, out innerMin, out innerMax);
            return innerMin.x < innerMax.x && innerMin.y < innerMax.y;
        }

        private static void ComputeInnerClampBox(float2 wMin, float2 wMax, float margin, out float2 innerMin, out float2 innerMax)
        {
            var m = BoundsUtility.EffectiveMargin(wMin, wMax, margin);
            innerMin = wMin + m;
            innerMax = wMax - m;
        }

        private static void RemapSpawnBufferToInnerRect(NativeList<SpawnParticle> buffer, float2 innerMin, float2 innerMax)
        {
            if (buffer.Length == 0)
                return;

            var pMin = buffer[0].position;
            var pMax = pMin;
            for (var i = 1; i < buffer.Length; i++)
            {
                var p = buffer[i].position;
                pMin = math.min(pMin, p);
                pMax = math.max(pMax, p);
            }

            var srcCenter = (pMin + pMax) * 0.5f;
            var srcExt = (pMax - pMin) * 0.5f;
            srcExt.x = math.max(srcExt.x, 1e-6f);
            srcExt.y = math.max(srcExt.y, 1e-6f);
            var dstCenter = (innerMin + innerMax) * 0.5f;
            var dstExt = (innerMax - innerMin) * 0.5f;
            var scale = math.min(dstExt.x / srcExt.x, dstExt.y / srcExt.y);
            for (var i = 0; i < buffer.Length; i++)
            {
                var sp = buffer[i];
                sp.position = dstCenter + (sp.position - srcCenter) * scale;
                buffer[i] = sp;
            }
        }
    }
}