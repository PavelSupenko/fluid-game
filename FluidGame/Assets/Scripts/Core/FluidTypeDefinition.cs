using UnityEngine;
using System;

/// <summary>
/// Defines properties for one type of fluid.
/// Density, viscosity, and cohesion will be used starting from Stage 2-4.
/// </summary>
[Serializable]
public struct FluidTypeDefinition
{
    public string name;
    public Color color;

    [Tooltip("Higher values = heavier fluid, sinks below lighter ones")]
    public float density;

    [Tooltip("Resistance to flow. Higher = more sluggish movement")]
    public float viscosity;

    [Tooltip("How strongly particles of this type attract each other")]
    public float cohesion;
}
