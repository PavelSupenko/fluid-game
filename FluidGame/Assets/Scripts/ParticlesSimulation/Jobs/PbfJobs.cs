using ParticlesSimulation.Components;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Jobs
{
    /// <summary>
    /// Computes SPH density for each particle using the Poly6 kernel over the spatial hash grid.
    /// Includes self-contribution (W(0) = Poly6Coefficient · h⁶).
    /// Operates on flat arrays extracted from ECS for cache-friendly access.
    /// </summary>
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    public struct ComputeDensityJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> Positions;
        [ReadOnly] public NativeParallelMultiHashMap<int, int> Grid;
        public float CellSizeInverse;
        public float SmoothingRadiusSq;
        public float Poly6Coefficient;
        public float ParticleMass;

        [WriteOnly] public NativeArray<float> Densities;

        public void Execute(int index)
        {
            var position = Positions[index];
            var cell = SpatialHash.CellCoords(position, CellSizeInverse);
            var smoothingRadiusSq = SmoothingRadiusSq;

            // Self-contribution: W_poly6(0, h) = Poly6Coefficient · (h²)³
            var selfTerm = smoothingRadiusSq * smoothingRadiusSq * smoothingRadiusSq;
            var density = ParticleMass * Poly6Coefficient * selfTerm;

            for (var dy = -1; dy <= 1; dy++)
            {
                for (var dx = -1; dx <= 1; dx++)
                {
                    var neighborCellHash = SpatialHash.Hash(cell + new int2(dx, dy));

                    if (!Grid.TryGetFirstValue(neighborCellHash, out var neighborIndex, out var iterator))
                        continue;

                    do
                    {
                        if (neighborIndex == index)
                            continue;

                        var distanceSq = math.lengthsq(Positions[neighborIndex] - position);
                        if (distanceSq >= smoothingRadiusSq)
                            continue;

                        // W_poly6(r, h) = Poly6Coefficient · (h² − r²)³
                        var difference = smoothingRadiusSq - distanceSq;
                        density += ParticleMass * Poly6Coefficient * difference * difference * difference;
                    } while (Grid.TryGetNextValue(out neighborIndex, ref iterator));
                }
            }

            Densities[index] = density;
        }
    }

    /// <summary>
    /// Computes the PBF Lagrange multiplier λ for each particle.
    /// λᵢ = −Cᵢ / (Σₖ |∇ₚₖ Cᵢ|² + ε), where Cᵢ = ρᵢ/ρ₀ − 1.
    /// Uses the Spiky kernel gradient for the constraint gradient.
    /// </summary>
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    public struct ComputeLambdaJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> Positions;
        [ReadOnly] public NativeArray<float> Densities;
        [ReadOnly] public NativeParallelMultiHashMap<int, int> Grid;
        public float CellSizeInverse;
        public float SmoothingRadius;
        public float SmoothingRadiusSq;
        public float SpikyGradCoefficient;
        public float RestDensity;
        public float ParticleMass;
        public float Epsilon;

        [WriteOnly] public NativeArray<float> Lambdas;

        public void Execute(int index)
        {
            var position = Positions[index];
            var density = Densities[index];
            var cell = SpatialHash.CellCoords(position, CellSizeInverse);
            var inverseRestDensity = math.rcp(RestDensity);

            // PBF constraint: C = ρ/ρ₀ − 1
            var constraint = density * inverseRestDensity - 1f;

            // Accumulate constraint gradient magnitude squared.
            // ∇ₚᵢ Cᵢ = (1/ρ₀) · Σⱼ ∇W_spiky(pᵢ − pⱼ)   (sum of all neighbor gradients)
            // ∇ₚⱼ Cᵢ = −(1/ρ₀) · ∇W_spiky(pᵢ − pⱼ)      (single neighbor gradient)
            var gradientSumI = float2.zero;
            var denominatorNeighbors = 0f;

            for (var dy = -1; dy <= 1; dy++)
            {
                for (var dx = -1; dx <= 1; dx++)
                {
                    var neighborCellHash = SpatialHash.Hash(cell + new int2(dx, dy));

                    if (!Grid.TryGetFirstValue(neighborCellHash, out var neighborIndex, out var iterator))
                        continue;

                    do
                    {
                        if (neighborIndex == index)
                            continue;

                        var toNeighbor = Positions[neighborIndex] - position;
                        var distanceSq = math.lengthsq(toNeighbor);
                        if (distanceSq >= SmoothingRadiusSq || distanceSq < 1e-12f)
                            continue;

                        var distance = math.sqrt(distanceSq);
                        var direction = toNeighbor / distance;

                        // ∇W_spiky = SpikyGradCoefficient · (h − r)² · r̂
                        var hMinusR = SmoothingRadius - distance;
                        var gradW = SpikyGradCoefficient * hMinusR * hMinusR * direction;

                        // ∇ₚⱼ Cᵢ = −(mass/ρ₀) · gradW
                        var gradJ = -ParticleMass * inverseRestDensity * gradW;
                        denominatorNeighbors += math.lengthsq(gradJ);

                        // Accumulate for ∇ₚᵢ Cᵢ
                        gradientSumI += ParticleMass * inverseRestDensity * gradW;
                    } while (Grid.TryGetNextValue(out neighborIndex, ref iterator));
                }
            }

            var denominator = math.lengthsq(gradientSumI) + denominatorNeighbors + Epsilon;
            Lambdas[index] = -constraint / denominator;
        }
    }

    /// <summary>
    /// Writes computed density and lambda from flat arrays back into <see cref="ParticleFluid"/> components.
    /// Entity query order must match the flat array indexing used by the solver.
    /// </summary>
    [BurstCompile]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct WriteBackFluidDataJob : IJobEntity
    {
        [ReadOnly] public NativeArray<float> Densities;
        [ReadOnly] public NativeArray<float> Lambdas;

        public void Execute([EntityIndexInQuery] int index, ref ParticleFluid fluid)
        {
            fluid.density = Densities[index];
            fluid.lambda = Lambdas[index];
        }
    }
}
