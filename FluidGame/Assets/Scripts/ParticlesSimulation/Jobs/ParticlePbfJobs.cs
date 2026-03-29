using ParticlesSimulation.Components;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Jobs
{
    /// <summary>
    /// Nine spatial hash keys for the 3×3 cell neighborhood. Blittable struct for Burst (Span is not valid in Burst indirect calls).
    /// </summary>
    internal struct NeighborCellHashes9
    {
        public int H0, H1, H2, H3, H4, H5, H6, H7, H8;

        public static NeighborCellHashes9 FromOrigin(int2 origin)
        {
            var x = origin.x;
            var y = origin.y;
            return new NeighborCellHashes9
            {
                H0 = SpatialHash2D.HashCell(x - 1, y - 1),
                H1 = SpatialHash2D.HashCell(x - 1, y),
                H2 = SpatialHash2D.HashCell(x - 1, y + 1),
                H3 = SpatialHash2D.HashCell(x, y - 1),
                H4 = SpatialHash2D.HashCell(x, y),
                H5 = SpatialHash2D.HashCell(x, y + 1),
                H6 = SpatialHash2D.HashCell(x + 1, y - 1),
                H7 = SpatialHash2D.HashCell(x + 1, y),
                H8 = SpatialHash2D.HashCell(x + 1, y + 1),
            };
        }

        public readonly int Get(int i) => i switch
        {
            0 => H0,
            1 => H1,
            2 => H2,
            3 => H3,
            4 => H4,
            5 => H5,
            6 => H6,
            7 => H7,
            8 => H8,
            _ => H4
        };
    }

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

    /// <summary>
    /// Clamps <see cref="ParticleCore.predictedPosition"/> to the simulation AABB. Run once per PBF substep after all
    /// solver iterations — not inside the iteration loop — so boundary clamping does not fight PBF separation.
    /// </summary>
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    [WithAll(typeof(ParticleSimTag))]
    internal partial struct ClampPredictedToWorldBoundsJob : IJobEntity
    {
        public SimulationWorldBounds WorldBounds;

        public void Execute(ref ParticleCore core)
        {
            if (WorldBounds.BoundsEnabled == 0)
                return;

            var ext = WorldBounds.Max - WorldBounds.Min;
            var margin = math.min(WorldBounds.Margin, math.max(0f, 0.49f * math.cmin(ext)));
            var min = WorldBounds.Min + margin;
            var max = WorldBounds.Max - margin;
            core.predictedPosition = math.clamp(core.predictedPosition, min, max);
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

            var neighborHashes = NeighborCellHashes9.FromOrigin(origin);

            for (var ni = 0; ni < 9; ni++)
            {
                var key = neighborHashes.Get(ni);
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

            var neighborHashes = NeighborCellHashes9.FromOrigin(origin);

            for (var ni = 0; ni < 9; ni++)
            {
                var key = neighborHashes.Get(ni);
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

            var neighborHashes = NeighborCellHashes9.FromOrigin(origin);

            for (var ni = 0; ni < 9; ni++)
            {
                var key = neighborHashes.Get(ni);
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