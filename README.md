# Fluid Simulation — Stage 1: Scaffold & Basic Particle System

## What This Stage Does
- Spawns a grid of colored particles (3 fluid types: red/green/blue)
- Particles fall under gravity
- Particles bounce off container walls with damping
- Rendered as smooth colored circles via GPU instancing
- Debug overlay shows particle count and FPS

## Project Structure

```
Assets/
├── FluidSim/
│   ├── Scripts/
│   │   ├── Core/
│   │   │   ├── FluidParticle.cs          # Particle data struct
│   │   │   ├── FluidTypeDefinition.cs    # Per-type properties
│   │   │   └── FluidSimulation.cs        # Main simulation loop
│   │   ├── Rendering/
│   │   │   └── FluidRenderer.cs          # GPU-instanced renderer
│   │   └── Debug/
│   │       └── FluidDebugOverlay.cs      # FPS + particle count
│   └── Shaders/
│       └── ParticleCircle.shader         # Instanced circle shader
```

## Setup Instructions

### 1. Create the folder structure
Copy all files into `Assets/FluidSim/` maintaining the folder structure above.

### 2. Create the Material
1. In Unity, right-click in `Assets/FluidSim/` → **Create → Material**
2. Name it `ParticleCircleMat`
3. Set its shader to **FluidSim/ParticleCircle** (find it in the shader dropdown)
4. Check **Enable GPU Instancing** in the material inspector

### 3. Set up the Scene
1. Create an **empty GameObject**, name it `FluidSimulation`
2. Add component: **FluidSimulation**
3. Add component: **FluidRenderer**
4. On the **FluidRenderer** component, drag `ParticleCircleMat` into the `Particle Material` slot
5. Add component: **FluidDebugOverlay** (on same or different object)

### 4. Camera Setup
1. Select the **Main Camera**
2. Set it to **Orthographic**
3. Set **Orthographic Size** to `5`
4. Position: `(0, 0.5, -10)`
5. Background color: dark gray or black for contrast

### 5. Hit Play
You should see ~600 colored particles falling and piling up at the bottom of the container.

## What to Test
- **Particles fall and bounce**: They should drop under gravity and settle at the bottom
- **Three colors visible**: Red (heavy), green (medium), blue (light) — stacked as horizontal bands
- **Container boundaries work**: No particles escape the yellow wireframe box (visible in Scene view)
- **FPS overlay**: Top-left corner shows particle count and framerate

## Tuning in the Inspector
- `Grid Width / Height` — increase for more particles (try 50x40 = 2000)
- `Particle Spacing` — tighter spacing = denser initial grid
- `Boundary Damping` — 0 = particles stop on contact, 1 = full elastic bounce
- `Time Scale` — slow down or speed up the sim
- `Render Scale` on FluidRenderer — size of each particle circle

## Next Stage (Stage 2)
SPH density & pressure forces so particles behave as a fluid instead of
independent projectiles. Spatial hashing for neighbor search.
