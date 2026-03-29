using ParticlesSimulation.Components;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Jobs
{
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimTag))]
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
    [WithAll(typeof(ParticleSimTag))]
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

    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal struct ClearSpatialHashJob : IJob
    {
        public NativeParallelMultiHashMap<int, Entity> map;

        public void Execute()
        {
            map.Clear();
        }
    }

    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimTag))]
    internal partial struct BuildSpatialHashJob : IJobEntity
    {
        public NativeParallelMultiHashMap<int, Entity>.ParallelWriter writer;
        public float cellInv;
        public bool usePredictedPositions;

        public void Execute(Entity entity, in ParticleCore core, ref GridHash gridHash)
        {
            var p = usePredictedPositions ? core.predictedPosition : core.position;
            var key = SpatialHash2D.HashPosition(p, cellInv);
            gridHash.cellHash = key;
            writer.Add(key, entity);
        }
    }

    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimTag))]
    internal partial struct ComputeDensityJob : IJobEntity
    {
        [ReadOnly] public NativeParallelMultiHashMap<int, Entity> grid;
        [ReadOnly] public ComponentLookup<ParticleCore> cores;

        public float cellInv;
        public float h2;
        public float poly6Coefficient;
        public float uniformParticleMass;

        public void Execute(Entity self, ref ParticleFluid fluid)
        {
            var pi = cores[self].predictedPosition;
            var density = 0f;
            var origin = SpatialHash2D.CellCoords(pi, cellInv);
            var mj = uniformParticleMass;

            for (var ox = -1; ox <= 1; ox++)
            {
                for (var oy = -1; oy <= 1; oy++)
                {
                    var key = SpatialHash2D.HashCell(origin.x + ox, origin.y + oy);
                    if (!grid.TryGetFirstValue(key, out var ej, out var iterator))
                        continue;

                    do
                    {
                        var pj = cores[ej].predictedPosition;
                        var d = pi - pj;
                        var r2 = math.lengthsq(d);
                        var w = PbfKernels.Poly6(r2, h2, poly6Coefficient);
                        density += w * mj;
                    } while (grid.TryGetNextValue(out ej, ref iterator));
                }
            }

            fluid.density = density;
            var rho0 = fluid.restDensity;
            var pressure = math.max(density / rho0 - 1f, 0f);
            fluid.pressure = pressure;
        }
    }

    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimTag))]
    internal partial struct ComputeLambdaJob : IJobEntity
    {
        [ReadOnly] public NativeParallelMultiHashMap<int, Entity> grid;
        [ReadOnly] public ComponentLookup<ParticleCore> cores;
        [ReadOnly] public ComponentLookup<ParticleState> states;

        public float cellInv;
        public float h;
        public float spikyGradCoefficient;
        public float epsilon;
        public float stiffness;
        public float uniformParticleMass;

        public void Execute(Entity self, ref ParticleFluid fluid)
        {
            var fluidMask = math.select(0f, 1f, states[self].phase == ParticlePhase.Fluid);
            var pi = cores[self].predictedPosition;
            var rho = fluid.density;
            var rho0 = fluid.restDensity;
            var denom = 0f;
            var origin = SpatialHash2D.CellCoords(pi, cellInv);
            var mj = uniformParticleMass;

            for (var ox = -1; ox <= 1; ox++)
            {
                for (var oy = -1; oy <= 1; oy++)
                {
                    var key = SpatialHash2D.HashCell(origin.x + ox, origin.y + oy);
                    if (!grid.TryGetFirstValue(key, out var ej, out var iterator))
                        continue;

                    do
                    {
                        var pj = cores[ej].predictedPosition;
                        var delta = pi - pj;
                        var r2 = math.lengthsq(delta);
                        var notSelf = math.select(0f, 1f, r2 > 1e-12f);
                        var fj = math.select(0f, 1f, states[ej].phase == ParticlePhase.Fluid);
                        PbfKernels.SpikyGradVec(delta, h, spikyGradCoefficient, out var g);
                        var gradScale = fluidMask * fj * notSelf;
                        var gradRho = g * mj;
                        denom += math.lengthsq(gradRho) * gradScale;
                    } while (grid.TryGetNextValue(out ej, ref iterator));
                }
            }

            var c = rho / rho0 - 1f;
            c = math.max(c, 0f);
            var lambda = -c * stiffness / (denom + epsilon);
            fluid.lambda = lambda * fluidMask;
        }
    }

    /// <summary>
    /// Reads only via lookups (no ref ParticleCore) so it does not alias with ComponentLookup&lt;ParticleCore&gt;.
    /// </summary>
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimTag))]
    internal partial struct AccumulatePbfDeltaJob : IJobEntity
    {
        [ReadOnly] public NativeParallelMultiHashMap<int, Entity> grid;
        [ReadOnly] public ComponentLookup<ParticleCore> cores;
        [ReadOnly] public ComponentLookup<ParticleFluid> fluids;
        [ReadOnly] public ComponentLookup<ParticleState> states;

        public NativeArray<float2> deltaOut;

        public float cellInv;
        public float h;
        public float spikyGradCoefficient;
        public float deltaScale;
        public float uniformParticleMass;

        public void Execute([EntityIndexInQuery] int entityIndexInQuery, Entity self)
        {
            var fluidMask = math.select(0f, 1f, states[self].phase == ParticlePhase.Fluid);
            var pi = cores[self].predictedPosition;
            var lambdaI = fluids[self].lambda;
            var rho0 = fluids[self].restDensity;
            var accum = float2.zero;
            var origin = SpatialHash2D.CellCoords(pi, cellInv);
            var mj = uniformParticleMass;

            for (var ox = -1; ox <= 1; ox++)
            {
                for (var oy = -1; oy <= 1; oy++)
                {
                    var key = SpatialHash2D.HashCell(origin.x + ox, origin.y + oy);
                    if (!grid.TryGetFirstValue(key, out var ej, out var iterator))
                        continue;

                    do
                    {
                        var pj = cores[ej].predictedPosition;
                        var delta = pi - pj;
                        var r2 = math.lengthsq(delta);
                        var notSelf = math.select(0f, 1f, r2 > 1e-12f);
                        var fj = math.select(0f, 1f, states[ej].phase == ParticlePhase.Fluid);
                        PbfKernels.SpikyGradVec(delta, h, spikyGradCoefficient, out var g);
                        var lambdaJ = fluids[ej].lambda;
                        var pair = fluidMask * fj * notSelf;
                        var coeff = (lambdaI + lambdaJ) / rho0 * mj * pair;
                        accum += coeff * g;
                    } while (grid.TryGetNextValue(out ej, ref iterator));
                }
            }

            deltaOut[entityIndexInQuery] = deltaScale * accum;
        }
    }

    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimTag))]
    internal partial struct ApplyPbfDeltaFromBufferJob : IJobEntity
    {
        [ReadOnly] public NativeArray<float2> deltaIn;

        public void Execute([EntityIndexInQuery] int entityIndexInQuery, ref ParticleCore core, in ParticleState state)
        {
            var fluidMask = math.select(0f, 1f, state.phase == ParticlePhase.Fluid);
            core.predictedPosition += deltaIn[entityIndexInQuery] * fluidMask;
        }
    }

    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimTag))]
    internal partial struct ApplyRigidShapeJob : IJobEntity
    {
        public float2 rigidCenter;
        public int rigidCount;
        public float shapeStiffness;

        public void Execute(ref ParticleCore core, in ParticleState state)
        {
            var hasRigid = math.select(0f, 1f, rigidCount > 0);
            var rigid = hasRigid * math.select(0f, 1f, state.phase == ParticlePhase.Rigid);
            var target = rigidCenter + state.initialLocalPosition;
            core.predictedPosition = math.lerp(core.predictedPosition, target, shapeStiffness * rigid);
        }
    }
}