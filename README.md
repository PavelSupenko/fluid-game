# Particle Simulation — Position Based Fluids (2D)

A 2D fluid simulation built on Unity ECS (Entities 1.x) and the Burst compiler, designed for mobile devices. Particles are spawned from an image (via color quantization) or a procedural grid and simulated as an incompressible viscous fluid using Position Based Fluids (PBF). Rendering is handled through Entities Graphics (URP).

---

## Project Structure

```
Assets/Game/Features/ParticlesSimulation/
│
├── Components/
│   └── SimulationComponents.cs        — ECS components and config singleton
│
├── Jobs/
│   ├── PbfJobs.cs                     — PBF solver jobs (density, lambda, corrections)
│   ├── SimulationJobs.cs              — Prediction, finalization, spawning jobs
│   └── SpatialHashJobs.cs             — Grid construction and neighbor counting
│
├── Rendering/
│   ├── ColorQuantizer.cs              — K-means palette extraction from images
│   ├── DebugParticleRenderController.cs — Runtime debug visualization
│   └── ImageToFluid.cs                — Image → particle buffer conversion
│
├── Systems/
│   ├── PbfDiagnosticSystem.cs         — Runtime density/velocity/lambda logging
│   ├── PbfSolverSystem.cs             — PBF constraint solver (iterative)
│   ├── SimulationSystems.cs           — Clock, prediction, finalization, transform sync
│   └── SpatialHashSystem.cs           — Spatial hash grid builder
│
├── ParticleSimulationBootstrap.cs     — MonoBehaviour entry point: spawning, config wiring
├── ParticleSimulationBounds.cs        — Container bounds (RectTransform-based)
├── ParticleSimulationGroup.cs         — ECS system group ordering
├── SimulationCheats.cs                — Debug commands and parameter overrides
└── SpatialHashing.cs                  — Static hash utilities (cell coords, hash function)
```

---

## Simulation Pipeline

Each frame, the simulation runs the following pipeline inside `ParticleSimulationGroup`, in strict order:

```
Clock → Prediction → Spatial Hash → PBF Solver → Finalization → [XSPH] → Transform Sync
```

### 1. Clock (`ParticleSimulationClockSystem`)

Locks `deltaTime` to a fixed 1/60s. At framerates below 60 fps the simulation slows down rather than taking larger timesteps, which preserves stability. This is intentional — a variable timestep would require re-tuning all solver parameters per frame.

### 2. Prediction (`PredictionSystem` → `PredictPositionsJob`)

Integrates external forces (gravity) and writes `predictedPosition`:

```
velocity += gravity × dt
predictedPosition = position + velocity × dt
```

Only fluid-phase particles are integrated. Rigid particles keep `predictedPosition = position` until their own system (not yet implemented) moves them.

### 3. Spatial Hash (`SpatialHashGridSystem`)

Builds a `NativeParallelMultiHashMap<int, int>` from predicted positions. Cell size equals the smoothing radius so that neighbor search requires only the 3×3 cell neighborhood. The hash function uses two large primes (73856093, 19349663) — standard in SPH/PBF literature.

The system exposes `Grid`, `Positions`, `ParticleCount`, and `FinalJobHandle` for downstream systems. Neighbor counting (`CountNeighborsJob`) is conditionally compiled — it runs only in `UNITY_EDITOR` or `DEVELOPMENT_BUILD` to save performance on release builds.

### 4. PBF Solver (`PbfSolverSystem`)

The core of the simulation. Runs N iterations of the following sub-steps:

**4a. Compute Density (`ComputeDensityJob`)**

SPH density estimation using the Poly6 kernel:

```
ρᵢ = Σⱼ m · W_poly6(|pᵢ − pⱼ|, h)
```

Includes self-contribution `W(0) = poly6Coeff · h⁶`. The 2D Poly6 coefficient is `4 / (π · h⁸)`.

**4b. Compute Lambda (`ComputeLambdaJob`)**

PBF Lagrange multiplier for the incompressibility constraint:

```
λᵢ = −C(ρᵢ) / (Σₖ |∇ₚₖ Cᵢ|² + ε)
```

where `C = ρ/ρ₀ − 1`. The constraint can be configured as one-sided (`max(0, C)`, original PBF) or bilateral (raw `C`, provides cohesion for contained fluids). The `cohesionStrength` parameter controls how strongly the tension side is scaled.

Gradient computation uses the Spiky kernel (`−30/(πh⁵) · (h−r)² · r̂`), which has a sharper peak than Poly6 and produces better repulsive gradients at close range.

**4c. Compute Position Correction (`ComputePositionCorrectionJob`)**

```
Δpᵢ = (1/ρ₀) · Σⱼ (λᵢ + λⱼ + s_corr) · ∇W_spiky(pᵢ − pⱼ)
```

Includes artificial pressure `s_corr = −k · (W_poly6(r) / W_poly6(Δq))^n` to prevent tensile instability (particle clumping at the surface).

**4d. Apply Correction (`ApplyPositionCorrectionJob`)**

Applies Δp scaled by `ω / N` (SOR omega divided by iteration count), with magnitude clamping and boundary enforcement. This is damped Jacobi iteration — each particle's correction is computed from the same snapshot of positions, then applied simultaneously.

### 5. Finalization (`FinalizationSystem`)

Derives velocity from the solver's corrected position:

```
velocity = (predicted − position) / dt
```

Applies per-frame displacement clamping (safety valve), boundary clamping with friction, and a hard speed cap. Then runs scalar fluid damping.

### 6. XSPH Viscosity (`XsphViscositySystem`, optional)

Blends each particle's velocity toward the weighted average of its neighbors:

```
v_new = v_old + c · Σⱼ (vⱼ − vᵢ) · W_poly6(rᵢⱼ) / ρⱼ
```

Produces coherent, viscous flow (ideal for thick fluids like honey or ice cream) without the energy-killing effect of scalar damping. Runs after finalization using the spatial hash grid from earlier in the frame.

### 7. Transform Sync (`ParticleLocalTransformSyncSystem`)

Copies `ParticleCore.position` into `LocalTransform.Position` for Entities Graphics rendering. Runs after the simulation group, before `TransformSystemGroup`.

---

## ECS Components

| Component | Purpose |
|---|---|
| `ParticleCore` | Position, predicted position, velocity, solver displacement (warm-start) |
| `ParticleFluid` | SPH density, rest density, lambda (solver scratch) |
| `ParticleState` | Phase (Fluid/Rigid) and palette color index |
| `ParticleSimulatedTag` | Query filter — marks active simulation particles |
| `ParticleOriginalColor` | Spawn-time color for debug visualization restore |
| `SimulationConfig` | Singleton — all solver parameters (gravity, kernel coefficients, etc.) |
| `SimulationWorldBounds` | Singleton — axis-aligned container with margin |
| `SpatialGridMapTag` | Singleton — marks the entity that holds config and bounds |

---

## Key Parameters

### Smoothing Radius (`_smoothingRadiusMultiplier`)

Controls how many neighbors each particle "sees". Defined as `h = spacing × multiplier`.

| Multiplier | Neighbors (2D) | Effect |
|---|---|---|
| 1.5 | ~8 | Minimum viable. Noisy density estimates. |
| 2.0 | ~12 | Standard. Good density accuracy. |
| 3.0 | ~28 | Smooth. More expensive per iteration. |
| 4.0+ | ~50+ | Very smooth but costly. May reduce needed iterations. |

Higher multiplier means each iteration is more expensive (more neighbors to visit), but pressure propagates further per iteration — so fewer iterations may be needed. There is a tradeoff between multiplier and iteration count.

### Solver Iterations (`_solverIterations` / `_autoIterations`)

Each iteration propagates pressure information approximately one kernel width. For a container that spans K kernel widths, approximately K/4 iterations are needed for full pressure propagation (with inter-frame accumulation).

With `_autoIterations` enabled, the bootstrap computes: `iterations = ceil(kernelWidths / (4 × ω))`, clamped to [3, 8].

### SOR Omega (`_sorOmega`)

Scales each iteration's correction by `ω / N`. With N iterations running in parallel Jacobi:

| Omega | Per-iteration factor (N=4) | Behavior |
|---|---|---|
| 0.5 | 0.125 | Very conservative. Slow convergence but very stable. |
| 0.7 | 0.175 | Sweet spot for stability. Recommended starting point. |
| 1.0 | 0.250 | Standard Jacobi. May oscillate with bilateral constraints. |
| 1.5+ | 0.375+ | Over-relaxation. Risk of divergence in parallel Jacobi. |

Values above ~1.0 tend to cause oscillation because Jacobi iterations don't account for simultaneous neighbor movement.

### Rest Density (`_autoEstimateRestDensity`)

When enabled, rest density is computed by analytically evaluating the Poly6 kernel over a regular grid with the measured particle spacing. This ensures `ρ₀` matches the actual initial packing. If the estimate is wrong, the solver either over-corrects (explosion) or under-corrects (collapse).

### Cohesion Strength (`_cohesionStrength`)

Scales the tension (under-dense) side of the bilateral constraint:

| Value | Behavior |
|---|---|
| 0.0 | One-sided constraint. No cohesion. Sharp transition at ρ₀. |
| 0.2 | Mild cohesion. Smooth transition, minimal volume collapse. Recommended. |
| 1.0 | Full symmetric. Strong surface tension effect. May over-compress. |

### Warm-Start Fraction (`_warmStartFraction`)

Reuses previous frame's solver displacement as initial guess. The solver starts near the converged solution and needs fewer iterations:

| Value | Effect |
|---|---|
| 0.0 | Cold start every frame. Needs many iterations. |
| 0.8–0.9 | Standard. Dramatic iteration reduction for settled fluid. |
| 1.0 | Full reuse. May accumulate drift over time. |

---

## Bootstrap Flow

`ParticleSimulationBootstrap` (MonoBehaviour, `Awake`) orchestrates initialization:

1. **Parse image** — `ImageToFluid.TryParseImage()` generates particle positions and colors from the source texture via color quantization.
2. **Build spawn buffer** — Collects `SpawnParticle` structs (position, color, type index). If an image source exists, particles are remapped to fit the simulation container. Otherwise, a procedural grid is generated.
3. **Measure spacing** — Samples nearest-neighbor distances from the buffer. This measured value is the single source of truth for all derived parameters.
4. **Derive parameters** — `smoothingRadius = spacing × multiplier`, `quadHalfExtent = spacing × 0.45`, rest density estimated from kernel evaluation at measured spacing.
5. **Create singletons** — `SimulationConfig`, `SimulationWorldBounds`, `SpatialGridMapTag` on a dedicated entity.
6. **Spawn entities** — Instantiates the particle archetype, runs `SetupParticlesJob` to write components, plays back the entity command buffer.

---

## Image-to-Fluid Pipeline

`ImageToFluid` converts a source texture into particle data:

1. **Sample** — Downsamples the image to `resolution × (resolution / aspect)` pixels.
2. **Quantize** — K-means color quantization reduces to `targetColorCount` palette entries. Colors below `minColorPercentage` are merged into their nearest major color.
3. **Position** — Each non-transparent pixel becomes a particle, positioned edge-to-edge within the simulation container (centered horizontally, bottom-aligned).
4. **Output** — `GeneratedParticles[]` (position + type index), `GeneratedFluidTypes[]` (palette colors), `ComputedSpacing`.

---

## Spatial Hashing

`SpatialHash` provides Burst-compatible 2D spatial hashing:

```
cell = (floor(x / h), floor(y / h))
hash = cell.x × 73856093 ^ cell.y × 19349663
```

Cell size equals smoothing radius, so the 3×3 cell neighborhood covers all particles within h. The `NativeParallelMultiHashMap` maps hash → particle index for O(1) neighbor lookup.

**Entity ordering**: All systems that use flat arrays indexed by `[EntityIndexInQuery]` (SpatialHash, PBF Solver, XSPH) must use identical entity queries (`WithAll<ParticleCore, ParticleFluid, ParticleState, ParticleSimulatedTag>`) to guarantee consistent indexing.

---

## Diagnostics

`PbfDiagnosticSystem` logs per-frame metrics after the simulation pipeline completes:

- **Density**: min, max, average, and ratio to ρ₀ (alerts if > 2×)
- **Lambda**: min/max (indicates constraint magnitude)
- **Speed**: max particle speed and index (alerts if > threshold)
- **Position**: bounding box of all particles

Logs the first 5 frames in detail, then every 10th frame, plus any frame that triggers an alert. Startup config is logged once on the first frame.

---

## Performance Notes

The simulation cost is dominated by the PBF solver: `particles × neighbors × 3 passes × iterations`. To optimize:

- Use `_smoothingRadiusMultiplier = 2.0` (12 neighbors) rather than higher values (28+ neighbors) when possible.
- Keep solver iterations at 3–4 with warm-starting enabled.
- `CountNeighborsJob` is stripped in release builds via `#if UNITY_EDITOR || DEVELOPMENT_BUILD`.
- All jobs are Burst-compiled with `FloatMode.Fast` and `FloatPrecision.Standard`.
- The spatial hash grid is built once per frame and reused across all solver iterations (valid because corrections are small enough that particles don't cross cell boundaries).

---

## Kernel Reference

The simulation uses two SPH kernels in 2D:

**Poly6** (density estimation, XSPH, artificial pressure):
```
W(r, h) = (4 / πh⁸) · (h² − r²)³     for r < h
```

**Spiky gradient** (lambda and correction gradients):
```
∇W(r, h) = (−30 / πh⁵) · (h − r)² · r̂    for r < h
```

Poly6 is smooth at r=0 (good for density), but its gradient vanishes there — so Spiky is used for repulsive forces, where the sharp peak at r→0 provides strong close-range repulsion.
