using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;

namespace ParticlesSimulation
{
    /// <summary>
    /// Spawns ECS particles from <see cref="ImageToFluid"/> output (Core) or a procedural grid,
    /// wires singleton simulation data, and initializes the shared spatial hash capacity.
    /// </summary>
    [DefaultExecutionOrder(-500)]
    public sealed class ParticleSimulationBootstrap : MonoBehaviour
    {
        [FormerlySerializedAs("imageToFluid")]
        [Header("Sources (Core)")]
        [SerializeField]
        private ImageToFluid _imageToFluid;

        [FormerlySerializedAs("fallbackGridX")]
        [Header("Fallback grid (no image)")]
        [SerializeField]
        private int _fallbackGridX = 32;
        [FormerlySerializedAs("fallbackGridY")] [SerializeField]
        private int _fallbackGridY = 48;
        [FormerlySerializedAs("fallbackOrigin")] [SerializeField]
        private float2 _fallbackOrigin = new float2(-1.6f, -2.2f);
        [FormerlySerializedAs("fallbackSpacing")] [SerializeField]
        private float _fallbackSpacing = 0.08f;

        [FormerlySerializedAs("smoothingRadius")]
        [Header("Simulation tuning")]
        [SerializeField]
        private float _smoothingRadius = 0.12f;
        [FormerlySerializedAs("meltLineY")] [SerializeField]
        private float _meltLineY = -1.2f;
        [FormerlySerializedAs("gravityY")] [SerializeField]
        private float _gravityY = -12f;
        [FormerlySerializedAs("viscosity")] [SerializeField]
        private float _viscosity = 0.35f;
        [FormerlySerializedAs("stiffness")] [SerializeField]
        private float _stiffness = 0.85f;
        [FormerlySerializedAs("rigidShapeStiffness")] [SerializeField]
        private float _rigidShapeStiffness = 0.65f;
        [FormerlySerializedAs("solverIterations")] [SerializeField]
        private int _solverIterations = 2;
        [FormerlySerializedAs("particleMass")] [SerializeField]
        private float _particleMass = 1f;
        [FormerlySerializedAs("restDensity")] [SerializeField]
        private float _restDensity = 1150f;

        [FormerlySerializedAs("dynamicRenderer")]
        [Header("Rendering hook")]
        [SerializeField]
        private ParticleDynamicQuadRenderer _dynamicRenderer;

        private EntityManager em;
        private Entity singletonEntity;
        private bool spawned;

        private void Awake()
        {
            if (!Application.isPlaying)
                return;

            var world = World.DefaultGameObjectInjectionWorld;
            if (world == null || !world.IsCreated)
            {
                Debug.LogError("[ParticleSimulationBootstrap] Default ECS world not available.");
                return;
            }

            em = world.EntityManager;
            CreateSingletons();
            SpawnParticles();
            ParticleSimulationSpatialGrid.EnsureCapacity(
                em.GetComponentData<SimulationConfig>(singletonEntity).maxParticles);

            if (_dynamicRenderer != null)
                _dynamicRenderer.Initialize(world);
        }

        private void OnDestroy()
        {
            ParticleSimulationSpatialGrid.DisposeAll();
        }

        private void CreateSingletons()
        {
            singletonEntity = em.CreateEntity();
            em.AddComponentData(singletonEntity, new SpatialGridMapTag());
            em.AddComponentData(singletonEntity, new RigidComState { center = float2.zero, count = 0 });

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

            var cfg = SimulationConfigUtility.CreateDefault(maxEstimate);
            cfg.gravityY = _gravityY;
            cfg.meltLineY = _meltLineY;
            cfg.viscosityMultiplier = _viscosity;
            cfg.stiffness = _stiffness;
            cfg.rigidShapeStiffness = _rigidShapeStiffness;
            cfg.solverIterations = _solverIterations;
            SimulationConfigUtility.ApplySmoothingRadius(ref cfg, _smoothingRadius);
            cfg.deltaTime = Time.fixedDeltaTime > 0f ? Time.fixedDeltaTime : Time.deltaTime;
            cfg.maxParticles = math.max(cfg.maxParticles, maxEstimate + 256);
            cfg.uniformParticleMass = _particleMass;

            em.AddComponentData(singletonEntity, cfg);
        }

        private void SpawnParticles()
        {
            if (spawned)
                return;

            var buffer = new NativeList<SpawnParticle>(Allocator.Temp);
            try
            {
                if (_imageToFluid != null && _imageToFluid.IsReady)
                    AppendFromImage(buffer);
                else
                    AppendFallbackGrid(buffer);

                if (buffer.Length == 0)
                {
                    Debug.LogWarning("[ParticleSimulationBootstrap] No particles spawned.");
                    return;
                }

                var com = float2.zero;
                for (var i = 0; i < buffer.Length; i++)
                    com += buffer[i].position;
                com /= buffer.Length;

                var archetype = em.CreateArchetype(
                    typeof(ParticleCore),
                    typeof(ParticleFluid),
                    typeof(ParticleState),
                    typeof(GridHash),
                    typeof(ParticleDrawColor),
                    typeof(ParticleSimTag));

                using var entities = new NativeArray<Entity>(buffer.Length, Allocator.Temp);
                em.CreateEntity(archetype, entities);

                for (var i = 0; i < buffer.Length; i++)
                {
                    var p = buffer[i];
                    var local = p.position - com;
                    var colorId = (byte)math.min(p.colorIndex, 7);

                    var core = new ParticleCore
                    {
                        position = p.position,
                        predictedPosition = p.position,
                        velocity = float2.zero,
                        mass = _particleMass
                    };

                    var fluid = new ParticleFluid
                    {
                        density = 0f,
                        pressure = 0f,
                        restDensity = _restDensity,
                        mass = _particleMass,
                        lambda = 0f
                    };

                    var st = new ParticleState
                    {
                        phase = ParticlePhase.Rigid,
                        initialLocalPosition = local,
                        colorId = colorId
                    };

                    var e = entities[i];
                    em.SetComponentData(e, core);
                    em.SetComponentData(e, fluid);
                    em.SetComponentData(e, st);
                    em.SetComponentData(e, default(GridHash));
                    em.SetComponentData(e, new ParticleDrawColor { value = p.color });
                }

                var cfg = em.GetComponentData<SimulationConfig>(singletonEntity);
                cfg.maxParticles = math.max(cfg.maxParticles, buffer.Length + 256);
                em.SetComponentData(singletonEntity, cfg);

                spawned = true;
            }
            finally
            {
                buffer.Dispose();
            }
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

        private struct SpawnParticle
        {
            public float2 position;
            public float4 color;
            public int colorIndex;
        }
    }
}
