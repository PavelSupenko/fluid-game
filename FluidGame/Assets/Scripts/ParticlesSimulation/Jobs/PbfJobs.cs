using ParticlesSimulation.Components;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;

namespace ParticlesSimulation.Jobs
{
    /// <summary>
    /// Copies predicted positions into the solver's working array.
    /// Keeps the spatial hash system's data untouched while the solver modifies positions.
    /// </summary>
    [BurstCompile]
    public struct CopyPositionsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> Source;
        [WriteOnly] public NativeArray<float2> Destination;

        public void Execute(int index)
        {
            Destination[index] = Source[index];
        }
    }

    /// <summary>
    /// Computes SPH density for each particle using the Poly6 kernel over the spatial hash grid.
    /// Includes self-contribution (W(0) = Poly6Coefficient · h⁶).
    /// Operates on the solver's working positions (updated each iteration).
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
            var hSq = SmoothingRadiusSq;

            // Self-contribution: W_poly6(0, h) = Poly6Coefficient · (h²)³
            var selfTerm = hSq * hSq * hSq;
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
                        if (distanceSq >= hSq)
                            continue;

                        // W_poly6(r, h) = Poly6Coefficient · (h² − r²)³
                        var diff = hSq - distanceSq;
                        density += ParticleMass * Poly6Coefficient * diff * diff * diff;
                    } while (Grid.TryGetNextValue(out neighborIndex, ref iterator));
                }
            }

            Densities[index] = density;
        }
    }

    /// <summary>
    /// Computes the PBF Lagrange multiplier λ for each particle.
    /// λᵢ = −Cᵢ / (Σₖ |∇ₚₖ Cᵢ|² + ε), where Cᵢ = ρᵢ/ρ₀ − 1.
    /// Uses a bilateral constraint (not one-sided): particles are pushed apart when
    /// compressed AND pulled together when under-dense. This provides natural cohesion
    /// for contained fluids and prevents the sharp transition from zero-correction to
    /// large-correction that causes oscillation with one-sided constraints.
    /// Gradient direction convention: (pᵢ − pⱼ), giving correct ∇_{pᵢ}W signs.
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

            // Bilateral constraint: C = ρ/ρ₀ − 1.
            // Positive when compressed (pushes apart), negative when under-dense (pulls together).
            // This gives smooth approach to equilibrium from both sides, preventing the
            // oscillation that one-sided max(0,...) causes when particles suddenly compress.
            var constraint = density * inverseRestDensity - 1f;

            // Accumulate |∇_{pₖ} Cᵢ|² for the denominator.
            var gradientSumSelf = float2.zero;
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

                        // Direction: pᵢ − pⱼ (from neighbor toward self)
                        var fromNeighbor = position - Positions[neighborIndex];
                        var distanceSq = math.lengthsq(fromNeighbor);
                        if (distanceSq >= SmoothingRadiusSq || distanceSq < 1e-12f)
                            continue;

                        var distance = math.sqrt(distanceSq);
                        var direction = fromNeighbor / distance;

                        // ∇_{pᵢ} W_spiky(pᵢ − pⱼ) = SpikyGradCoeff · (h − r)² · (pᵢ − pⱼ)/r
                        var hMinusR = SmoothingRadius - distance;
                        var gradW = SpikyGradCoefficient * hMinusR * hMinusR * direction;

                        // ∇_{pⱼ} Cᵢ = −(m/ρ₀) · ∇_{pᵢ}W
                        var gradNeighbor = -ParticleMass * inverseRestDensity * gradW;
                        denominatorNeighbors += math.lengthsq(gradNeighbor);

                        // Accumulate ∇_{pᵢ} Cᵢ = (1/ρ₀) · Σⱼ m · ∇_{pᵢ}W
                        gradientSumSelf += ParticleMass * inverseRestDensity * gradW;
                    } while (Grid.TryGetNextValue(out neighborIndex, ref iterator));
                }
            }

            var denominator = math.lengthsq(gradientSumSelf) + denominatorNeighbors + Epsilon;
            Lambdas[index] = -constraint / denominator;
        }
    }

    /// <summary>
    /// Computes position correction Δp for each particle from the PBF constraint.
    /// Δpᵢ = (1/ρ₀) · Σⱼ (λᵢ + λⱼ + s_corr) · ∇W_spiky(pᵢ − pⱼ).
    /// Includes artificial pressure (s_corr) to prevent tensile instability.
    /// </summary>
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    public struct ComputePositionCorrectionJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> Positions;
        [ReadOnly] public NativeArray<float> Lambdas;
        [ReadOnly] public NativeParallelMultiHashMap<int, int> Grid;
        public float CellSizeInverse;
        public float SmoothingRadius;
        public float SmoothingRadiusSq;
        public float SpikyGradCoefficient;
        public float Poly6Coefficient;
        public float RestDensity;

        // Artificial pressure parameters (s_corr = −k · (W_poly6(r)/W_poly6(Δq))^n)
        public float ArtificialPressureStrength;
        public float ArtificialPressureExponent;
        /// <summary>Precomputed 1/W_poly6(Δq·h) for the artificial pressure ratio.</summary>
        public float InverseReferencePoly6;

        [WriteOnly] public NativeArray<float2> Corrections;

        public void Execute(int index)
        {
            var position = Positions[index];
            var lambdaSelf = Lambdas[index];
            var cell = SpatialHash.CellCoords(position, CellSizeInverse);
            var hSq = SmoothingRadiusSq;
            var correction = float2.zero;

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

                        var fromNeighbor = position - Positions[neighborIndex];
                        var distanceSq = math.lengthsq(fromNeighbor);
                        if (distanceSq >= hSq || distanceSq < 1e-12f)
                            continue;

                        var distance = math.sqrt(distanceSq);
                        var direction = fromNeighbor / distance;

                        // Spiky gradient: ∇_{pᵢ}W(pᵢ − pⱼ)
                        var hMinusR = SmoothingRadius - distance;
                        var gradW = SpikyGradCoefficient * hMinusR * hMinusR * direction;

                        // Artificial pressure: s_corr = −k · (W_poly6(r) / W_poly6(Δq))^n
                        var sCorr = 0f;
                        if (ArtificialPressureStrength > 0f)
                        {
                            var poly6Diff = hSq - distanceSq;
                            var wPoly6 = Poly6Coefficient * poly6Diff * poly6Diff * poly6Diff;
                            var ratio = wPoly6 * InverseReferencePoly6;
                            sCorr = -ArtificialPressureStrength
                                    * math.pow(ratio, ArtificialPressureExponent);
                        }

                        correction += (lambdaSelf + Lambdas[neighborIndex] + sCorr) * gradW;
                    } while (Grid.TryGetNextValue(out neighborIndex, ref iterator));
                }
            }

            Corrections[index] = correction * math.rcp(RestDensity);
        }
    }

    /// <summary>
    /// Applies position corrections with SOR (Successive Over-Relaxation), magnitude clamping,
    /// and boundary enforcement. SOR with ω > 1 overshoots corrections to accelerate
    /// convergence, effectively doubling the iteration count at ω ≈ 1.5.
    /// Boundary clamping during solver iteration prevents corrections from pushing
    /// particles outside the container.
    /// </summary>
    [BurstCompile]
    public struct ApplyPositionCorrectionJob : IJobParallelFor
    {
        public NativeArray<float2> Positions;
        [ReadOnly] public NativeArray<float2> Corrections;
        /// <summary>SOR relaxation factor ω (1.0 = standard Jacobi, 1.3–1.7 = accelerated).</summary>
        public float Omega;
        /// <summary>Maximum correction magnitude (typically fraction of smoothing radius).</summary>
        public float MaxCorrection;
        /// <summary>Clamping bounds (already margin-adjusted). Zero = disabled.</summary>
        public float2 BoundsMin;
        public float2 BoundsMax;
        public byte BoundsEnabled;

        public void Execute(int index)
        {
            var correction = Corrections[index] * Omega;
            var magnitudeSq = math.lengthsq(correction);
            if (magnitudeSq > MaxCorrection * MaxCorrection)
                correction *= MaxCorrection * math.rsqrt(magnitudeSq);

            var position = Positions[index] + correction;

            if (BoundsEnabled != 0)
                position = math.clamp(position, BoundsMin, BoundsMax);

            Positions[index] = position;
        }
    }

    /// <summary>
    /// Writes solver results (corrected positions, density, lambda) back into ECS components.
    /// Entity query order must match the flat array indexing used throughout the solver.
    /// </summary>
    [BurstCompile]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct WriteBackSolverResultsJob : IJobEntity
    {
        [ReadOnly] public NativeArray<float2> Positions;
        [ReadOnly] public NativeArray<float> Densities;
        [ReadOnly] public NativeArray<float> Lambdas;

        public void Execute([EntityIndexInQuery] int index, ref ParticleCore core, ref ParticleFluid fluid)
        {
            core.predictedPosition = Positions[index];
            fluid.density = Densities[index];
            fluid.lambda = Lambdas[index];
        }
    }

    /// <summary>
    /// XSPH velocity smoothing: blends each particle's velocity toward the weighted
    /// average of its neighbors. Produces coherent, viscous flow (ideal for thick
    /// fluids like honey or ice cream) without the energy-killing effect of scalar damping.
    /// Formula: v_new = v_old + c · Σⱼ (vⱼ − vᵢ) · W_poly6(rᵢⱼ) / ρⱼ
    /// </summary>
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    public struct XsphViscosityJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> Positions;
        [ReadOnly] public NativeArray<float2> Velocities;
        [ReadOnly] public NativeArray<float> Densities;
        [ReadOnly] public NativeParallelMultiHashMap<int, int> Grid;
        public float CellSizeInverse;
        public float SmoothingRadiusSq;
        public float Poly6Coefficient;
        /// <summary>XSPH blending factor c (0..1).</summary>
        public float Viscosity;

        [WriteOnly] public NativeArray<float2> SmoothedVelocities;

        public void Execute(int index)
        {
            var position = Positions[index];
            var velocity = Velocities[index];
            var cell = SpatialHash.CellCoords(position, CellSizeInverse);
            var hSq = SmoothingRadiusSq;
            var delta = float2.zero;

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
                        if (distanceSq >= hSq)
                            continue;

                        var diff = hSq - distanceSq;
                        var wPoly6 = Poly6Coefficient * diff * diff * diff;

                        // Weight by 1/ρⱼ so denser regions don't dominate the average.
                        var neighborDensity = Densities[neighborIndex];
                        var weight = wPoly6 * math.rcp(math.max(neighborDensity, 1e-6f));

                        delta += (Velocities[neighborIndex] - velocity) * weight;
                    } while (Grid.TryGetNextValue(out neighborIndex, ref iterator));
                }
            }

            SmoothedVelocities[index] = velocity + delta * Viscosity;
        }
    }

    /// <summary>
    /// Writes XSPH-smoothed velocities back into ECS particle components.
    /// Must run after <see cref="XsphViscosityJob"/> fills the smoothed array.
    /// </summary>
    [BurstCompile]
    [WithAll(typeof(ParticleSimulatedTag))]
    internal partial struct WriteBackXsphVelocitiesJob : IJobEntity
    {
        [ReadOnly] public NativeArray<float2> SmoothedVelocities;

        public void Execute([EntityIndexInQuery] int index, ref ParticleCore core)
        {
            core.velocity = SmoothedVelocities[index];
        }
    }
}