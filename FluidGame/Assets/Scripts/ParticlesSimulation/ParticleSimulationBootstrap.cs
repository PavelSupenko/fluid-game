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

        [Header("Particle grid")]
        [SerializeField]
        private int _gridX = 32;

        [SerializeField]
        private int _gridY = 48;

        [Tooltip("Manual particle spacing (used when Auto Fit is off or no bounds assigned).")]
        [SerializeField]
        private float _particleSpacing = 0.08f;

        [Tooltip("Smoothing radius as a multiple of measured particle spacing. " +
                 "2.0 gives ~12 neighbors in 2D (stable). 1.5 gives ~8 (minimum). " +
                 "Higher = smoother density estimates but slower.")]
        [Range(1.5f, 10.0f)]
        [SerializeField]
        private float _smoothingRadiusMultiplier = 2.0f;

        [Tooltip("Spawn origin when no Simulation Bounds is assigned.")]
        [SerializeField]
        private float2 _fallbackOrigin = new float2(-1.6f, -2.2f);

        [Header("Simulation tuning")]
        [SerializeField]
        private float _smoothingRadius = 0.12f;

        [SerializeField]
        private float _gravityY = -9.81f;
        
        [SerializeField]
        private float _maxSpeed = 8f;

        [Tooltip("Per-frame scalar velocity damping for fluid particles (0 = no damping, 1 = full stop). " +
                 "Keep low (0.01–0.1) when using XSPH viscosity — XSPH handles viscous behavior better.")]
        [Range(0f, 1f)]
        [SerializeField]
        private float _fluidDamping = 0.05f;

        [Tooltip("PBD stiffness applied to position corrections (0..1). " +
                 "With SOR enabled, use 1.0 and let SOR omega control convergence rate.")]
        [Range(0f, 1f)]
        [SerializeField]
        private float _stiffness = 1f;

        [SerializeField]
        private int _solverIterations = 4;

        [SerializeField]
        private float _particleMass = 1f;

        [Tooltip("SOR factor divided by iteration count. 1.0 with 4 iterations = 0.25 per iteration. " +
                 "Increase to 1.5 for faster convergence. Range: 0.5–1.9.")]
        [Range(0.5f, 1.9f)]
        [SerializeField]
        private float _sorOmega = 1.0f;

        [Tooltip("XSPH velocity smoothing (0 = off, 0.3 = viscous, 0.6+ = very thick). " +
                 "Blends velocities between neighbors for cohesive flow. " +
                 "Better than scalar damping for thick fluids like ice cream.")]
        [Range(0f, 1f)]
        [SerializeField]
        private float _xsphViscosity = 0.3f;

        [Tooltip("Tangential friction at boundaries (0 = frictionless, 1 = full stop). " +
                 "Controls how particles slide along walls. 0.3 feels natural for most fluids.")]
        [Range(0f, 1f)]
        [SerializeField]
        private float _boundaryFriction = 0.3f;

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

        [Tooltip("Manual quad half extent. Ignored when Auto Fit To Container is enabled.")]
        [SerializeField]
        private float _quadHalfExtent = 0.035f;

        [SerializeField]
        private int _renderingLayer;

        private EntityManager entityManager;
        private Entity singletonEntity;
        private bool spawned;
        private Mesh _runtimeQuadMesh;

        // Resolved at Awake — may be auto-computed from container dimensions.
        private float _resolvedSpacing;
        private float _resolvedQuadHalfExtent;
        private float _resolvedSmoothingRadius;

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

            // Phase 1: Parse image source (if any) so we know particle count.
            ParseImageSource();

            // Phase 2: Build spawn buffer — actual particle positions are determined here.
            var buffer = BuildSpawnBuffer();
            if (buffer.Length == 0)
            {
                UnityEngine.Debug.LogWarning("[ParticleSimulationBootstrap] No particles to spawn.");
                buffer.Dispose();
                return;
            }

            // Phase 3: Measure actual nearest-neighbor distance from the buffer.
            //          This is the ONLY source of truth for spacing — no estimates.
            var measuredSpacing = MeasureActualSpacing(buffer);

            // Phase 4: Derive smoothing radius and visual size from measured spacing.
            _resolvedSpacing = measuredSpacing;
            _resolvedSmoothingRadius = measuredSpacing * _smoothingRadiusMultiplier;
            _resolvedQuadHalfExtent = measuredSpacing * 0.45f;

            UnityEngine.Debug.Log(
                $"[ParticleSimulationBootstrap] Measured spacing={_resolvedSpacing:F5}, " +
                $"smoothingRadius={_resolvedSmoothingRadius:F4} (×{_smoothingRadiusMultiplier:F2}), " +
                $"quadHalfExtent={_resolvedQuadHalfExtent:F4}, particles={buffer.Length}");

            // Phase 5: Create singleton entities with config derived from measured spacing.
            CreateSingletons(buffer.Length);

            // Phase 6: Instantiate particle entities from the buffer.
            SpawnEntities(buffer);

            buffer.Dispose();
        }

        /// <summary>
        /// Parses the image source if available. Must be called before BuildSpawnBuffer
        /// so that ImageToFluid has generated its particle data.
        /// </summary>
        private void ParseImageSource()
        {
            if (_imageToFluid == null || _simulationBounds == null)
                return;

            if (!_simulationBounds.TryGetWorldAabb(out var min, out var max))
                return;

            _imageToFluid.TryParseImage(Rect.MinMaxRect(min.x, min.y, max.x, max.y));
        }

        /// <summary>
        /// Builds the spawn buffer with final world-space particle positions.
        /// Positions come from either the image source or a procedural grid,
        /// and are remapped to fit the simulation container if bounds exist.
        /// </summary>
        private NativeList<SpawnParticle> BuildSpawnBuffer()
        {
            var buffer = new NativeList<SpawnParticle>(Allocator.TempJob);

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

            return buffer;
        }

        /// <summary>
        /// Measures the actual nearest-neighbor distance from a sample of the spawn buffer.
        /// This is the ground truth for particle spacing — everything else (smoothingRadius,
        /// restDensity, visual size) must be derived from this, not from grid estimates.
        /// </summary>
        private float MeasureActualSpacing(NativeList<SpawnParticle> buffer)
        {
            if (buffer.Length < 2)
                return _particleSpacing;

            // Sample up to 64 particles distributed evenly through the buffer.
            var sampleCount = math.min(buffer.Length, 64);
            var step = math.max(1, buffer.Length / sampleCount);
            var distSum = 0f;
            var samples = 0;

            for (var i = 0; i < buffer.Length && samples < sampleCount; i += step)
            {
                var pos = buffer[i].position;
                var nearest = float.MaxValue;

                // Brute-force nearest neighbor (only runs once at startup, not hot path).
                for (var j = 0; j < buffer.Length; j++)
                {
                    if (j == i) continue;
                    var d = math.lengthsq(buffer[j].position - pos);
                    if (d < nearest) nearest = d;
                }

                distSum += math.sqrt(nearest);
                samples++;
            }

            var measured = distSum / math.max(1, samples);

            // Sanity check: if measurement is wildly off, fall back to manual spacing.
            if (measured < 1e-6f || float.IsNaN(measured))
            {
                UnityEngine.Debug.LogWarning(
                    $"[ParticleSimulationBootstrap] Measured spacing is invalid ({measured:E2}), " +
                    $"falling back to manual spacing={_particleSpacing}");
                return _particleSpacing;
            }

            return measured;
        }

        private void OnDestroy()
        {
            if (_runtimeQuadMesh != null)
                Destroy(_runtimeQuadMesh);
        }

        private void CreateSingletons(int particleCount)
        {
            singletonEntity = entityManager.CreateEntity();
            entityManager.AddComponentData(singletonEntity, new SpatialGridMapTag());

            var cfg = ConfigUtility.CreateDefault(particleCount, _resolvedSmoothingRadius);
            cfg.gravityY = _gravityY;
            cfg.maxSpeed = _maxSpeed;
            cfg.fluidDamping = _fluidDamping;
            cfg.stiffness = _stiffness;
            cfg.solverIterations = _solverIterations;
            cfg.deltaTime = 1f / 60f;
            cfg.maxParticles = math.max(cfg.maxParticles, particleCount + 256);
            cfg.uniformParticleMass = _particleMass;
            cfg.sorOmega = _sorOmega;
            cfg.xsphViscosity = _xsphViscosity;
            cfg.boundaryFriction = _boundaryFriction;
            if (_autoEstimateRestDensity)
            {
                cfg.restDensity = ConfigUtility.EstimateRestDensity(in cfg, _resolvedSpacing);
                UnityEngine.Debug.Log($"[ParticleSimulationBootstrap] Auto-estimated restDensity = {cfg.restDensity:F1} " +
                                      $"(spacing={_resolvedSpacing:F5}, h={_resolvedSmoothingRadius:F4})");
            }
            else
            {
                cfg.restDensity = _restDensity;
            }

            entityManager.AddComponentData(singletonEntity, cfg);

            var worldBounds = new SimulationWorldBounds { BoundsEnabled = 0 };
            if (_simulationBounds != null && _simulationBounds.TryGetWorldAabb(out var wMin, out var wMax))
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

        /// <summary>
        /// Creates ECS entities from the pre-built spawn buffer. Config singleton must already exist.
        /// </summary>
        private void SpawnEntities(NativeList<SpawnParticle> buffer)
        {
            if (spawned || buffer.Length == 0)
                return;

            if (_particleMaterial == null)
            {
                UnityEngine.Debug.LogError(
                    "[ParticleSimulationBootstrap] No particle material assigned - cannot spawn entities with graphics components.");
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

            var spawnConfig = entityManager.GetComponentData<SimulationConfig>(singletonEntity);

            var setupJob = new SetupParticlesJob
            {
                Entities = entities,
                Buffer = buffer.AsArray(),
                CommandBuffer = commandBuffer.AsParallelWriter(),
                CenterOfMass = centerOfMass,
                RestDensity = spawnConfig.restDensity,
                QuadScale = _resolvedQuadHalfExtent * 2f,
                PositionJitter = 0f,// _resolvedSpacing * 0.02f,
                RandomSeed = 42u
            };

            setupJob.Schedule(buffer.Length, 64).Complete();
            commandBuffer.Playback(entityManager);

            var cfg = entityManager.GetComponentData<SimulationConfig>(singletonEntity);
            cfg.maxParticles = math.max(cfg.maxParticles, buffer.Length + 256);
            entityManager.SetComponentData(singletonEntity, cfg);

            spawned = true;
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
                var nx = math.max(1, _gridX - 1);
                var ny = math.max(1, _gridY - 1);
                var spacingX = size.x / nx;
                var spacingY = size.y / ny;
                for (var y = 0; y < _gridY; y++)
                {
                    for (var x = 0; x < _gridX; x++)
                    {
                        var pos = innerMin + new float2(x * spacingX, y * spacingY);
                        var t = (float)(x + y) / math.max(1, _gridX + _gridY - 2);
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

            for (var y = 0; y < _gridY; y++)
            {
                for (var x = 0; x < _gridX; x++)
                {
                    var pos = _fallbackOrigin + new float2(x * _particleSpacing, y * _particleSpacing);
                    var t = (float)(x + y) / math.max(1, _gridX + _gridY - 2);
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
            if (_simulationBounds == null)
                return false;
            if (!_simulationBounds.TryGetWorldAabb(out var wMin, out var wMax))
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