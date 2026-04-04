using ParticlesSimulation.Components;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Systems
{
    /// <summary>
    /// Diagnostic system that runs after the full simulation pipeline and logs
    /// key metrics to the console. Helps identify the root cause of instability
    /// by showing exact density/velocity/correction values at each frame.
    ///
    /// Enable by adding this system to your world (it auto-creates via [UpdateInGroup]).
    /// Disable by removing the file or toggling <see cref="Enabled"/> in a debugger.
    ///
    /// Output format (one line per frame):
    /// [PBF Diag] F=0 N=625 ρ=[280.3..412.7 avg=305.2] ρ₀=300.0 λ=[-0.02..0.00] |v|max=1.23 ...
    /// </summary>
    [UpdateInGroup(typeof(SimulationSystemGroup))]
    [UpdateAfter(typeof(ParticleSimulationGroup))]
    [DisableAutoCreation]
    public partial class PbfDiagnosticSystem : SystemBase
    {
        private EntityQuery _particleQuery;
        private int _frameCount;
        private bool _startupLogged;

        /// <summary>Log every N frames to avoid console spam. Set to 1 for detailed tracing.</summary>
        private const int LogInterval = 10;

        /// <summary>Always log the first N frames in detail regardless of LogInterval.</summary>
        private const int DetailedStartupFrames = 5;

        /// <summary>Velocity threshold that triggers an alert log.</summary>
        private const float AlertVelocity = 5f;

        /// <summary>Density ratio (ρ/ρ₀) threshold that triggers an alert log.</summary>
        private const float AlertDensityRatio = 2f;

        protected override void OnCreate()
        {
            _particleQuery = SystemAPI.QueryBuilder()
                .WithAll<ParticleCore, ParticleFluid, ParticleState, ParticleSimulatedTag>()
                .Build();

            RequireForUpdate(_particleQuery);
            RequireForUpdate<SimulationConfig>();
        }

        protected override void OnUpdate()
        {
            var config = SystemAPI.GetSingleton<SimulationConfig>();
            var particleCount = _particleQuery.CalculateEntityCount();

            if (particleCount == 0)
                return;

            // Allocate temporary arrays for reduction.
            var densities = new NativeArray<float>(particleCount, Allocator.TempJob);
            var lambdas = new NativeArray<float>(particleCount, Allocator.TempJob);
            var velocitySqs = new NativeArray<float>(particleCount, Allocator.TempJob);
            var positions = new NativeArray<float2>(particleCount, Allocator.TempJob);

            // Extract data from ECS.
            var extractHandle = new ExtractDiagnosticsJob
            {
                Densities = densities,
                Lambdas = lambdas,
                VelocitySqs = velocitySqs,
                Positions = positions
            }.ScheduleParallel(_particleQuery, Dependency);

            // Reduce to min/max/sum on a single thread.
            var result = new NativeReference<DiagnosticResult>(Allocator.TempJob);
            var reduceHandle = new ReduceDiagnosticsJob
            {
                Densities = densities,
                Lambdas = lambdas,
                VelocitySqs = velocitySqs,
                Positions = positions,
                ParticleCount = particleCount,
                Result = result
            }.Schedule(extractHandle);

            reduceHandle.Complete();

            var r = result.Value;
            var restDensity = config.restDensity;
            var maxSpeed = math.sqrt(r.maxVelocitySq);
            var avgDensity = r.densitySum / math.max(1, particleCount);
            var maxDensityRatio = r.maxDensity / math.max(1e-6f, restDensity);

            // Startup detailed report: log config + initial state.
            if (!_startupLogged)
            {
                _startupLogged = true;
                UnityEngine.Debug.Log(
                    $"[PBF Diag] === STARTUP CONFIG ===\n" +
                    $"  particles={particleCount}, h={config.smoothingRadius:F4}, h²={config.smoothingRadiusSq:F6}\n" +
                    $"  restDensity={restDensity:F2}, epsilon={config.pbfEpsilon:F2}, mass={config.uniformParticleMass:F2}\n" +
                    $"  stiffness={config.stiffness:F2}, SOR ω={config.sorOmega:F2}, iterations={config.solverIterations}\n" +
                    $"  maxSpeed={config.maxSpeed:F2}, maxCorrFrac={config.maxCorrectionFraction:F2}, maxDispFrac={config.maxDisplacementFraction:F2}\n" +
                    $"  gravity={config.gravityY:F2}, fluidDamping={config.fluidDamping:F3}, xsphViscosity={config.xsphViscosity:F2}\n" +
                    $"  poly6Coeff={config.poly6Coefficient:E4}, spikyGradCoeff={config.spikyGradCoefficient:E4}\n" +
                    $"  artificialPressure: k={config.artificialPressureStrength:F3}, n={config.artificialPressureExponent:F1}, dq/h={config.artificialPressureRadius:F2}\n" +
                    $"  boundaryFriction={config.boundaryFriction:F2}");
            }

            // Determine whether to log this frame.
            var isStartup = _frameCount < DetailedStartupFrames;
            var isAlertFrame = maxSpeed > AlertVelocity || maxDensityRatio > AlertDensityRatio;
            var isLogFrame = (_frameCount % LogInterval == 0);

            if (isStartup || isAlertFrame || isLogFrame)
            {
                var prefix = isAlertFrame && !isStartup ? "⚠ ALERT" : "      ";

                UnityEngine.Debug.Log(
                    $"[PBF Diag] {prefix} F={_frameCount} N={particleCount}\n" +
                    $"  density: min={r.minDensity:F2} max={r.maxDensity:F2} avg={avgDensity:F2} " +
                    $"(ρ₀={restDensity:F2}, max/ρ₀={maxDensityRatio:F3})\n" +
                    $"  lambda:  min={r.minLambda:F6} max={r.maxLambda:F6}\n" +
                    $"  speed:   max={maxSpeed:F4} (cap={config.maxSpeed:F2})\n" +
                    $"  pos:     min=({r.posMin.x:F3},{r.posMin.y:F3}) max=({r.posMax.x:F3},{r.posMax.y:F3})\n" +
                    $"  fastest particle idx={r.maxVelocityIndex}");
            }

            _frameCount++;

            densities.Dispose();
            lambdas.Dispose();
            velocitySqs.Dispose();
            positions.Dispose();
            result.Dispose();

            Dependency = default;
        }

        private struct DiagnosticResult
        {
            public float minDensity;
            public float maxDensity;
            public float densitySum;
            public float minLambda;
            public float maxLambda;
            public float maxVelocitySq;
            public int maxVelocityIndex;
            public float2 posMin;
            public float2 posMax;
        }

        [BurstCompile]
        [WithAll(typeof(ParticleSimulatedTag))]
        private partial struct ExtractDiagnosticsJob : IJobEntity
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<float> Densities;
            [NativeDisableParallelForRestriction]
            public NativeArray<float> Lambdas;
            [NativeDisableParallelForRestriction]
            public NativeArray<float> VelocitySqs;
            [NativeDisableParallelForRestriction]
            public NativeArray<float2> Positions;

            public void Execute(
                [EntityIndexInQuery] int index,
                in ParticleCore core,
                in ParticleFluid fluid)
            {
                Densities[index] = fluid.density;
                Lambdas[index] = fluid.lambda;
                VelocitySqs[index] = math.lengthsq(core.velocity);
                Positions[index] = core.position;
            }
        }

        [BurstCompile]
        private struct ReduceDiagnosticsJob : IJob
        {
            [ReadOnly] public NativeArray<float> Densities;
            [ReadOnly] public NativeArray<float> Lambdas;
            [ReadOnly] public NativeArray<float> VelocitySqs;
            [ReadOnly] public NativeArray<float2> Positions;
            public int ParticleCount;
            public NativeReference<DiagnosticResult> Result;

            public void Execute()
            {
                var r = new DiagnosticResult
                {
                    minDensity = float.MaxValue,
                    maxDensity = float.MinValue,
                    densitySum = 0f,
                    minLambda = float.MaxValue,
                    maxLambda = float.MinValue,
                    maxVelocitySq = 0f,
                    maxVelocityIndex = 0,
                    posMin = new float2(float.MaxValue),
                    posMax = new float2(float.MinValue)
                };

                for (var i = 0; i < ParticleCount; i++)
                {
                    var d = Densities[i];
                    r.minDensity = math.min(r.minDensity, d);
                    r.maxDensity = math.max(r.maxDensity, d);
                    r.densitySum += d;

                    var l = Lambdas[i];
                    r.minLambda = math.min(r.minLambda, l);
                    r.maxLambda = math.max(r.maxLambda, l);

                    var vs = VelocitySqs[i];
                    if (vs > r.maxVelocitySq)
                    {
                        r.maxVelocitySq = vs;
                        r.maxVelocityIndex = i;
                    }

                    var p = Positions[i];
                    r.posMin = math.min(r.posMin, p);
                    r.posMax = math.max(r.posMax, p);
                }

                Result.Value = r;
            }
        }
    }
}
