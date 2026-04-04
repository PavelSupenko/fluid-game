using ParticlesSimulation.Components;
using Unity.Entities;
using Unity.Mathematics;
using UnityEngine;

namespace ParticlesSimulation
{
    /// <summary>
    /// Drives <see cref="SimulationWorldBounds"/> from a <see cref="RectTransform"/> (UI or world-space).
    /// World corners are projected to XY for the 2D fluid sim. Assign the same component from
    /// <see cref="ParticleSimulationBootstrap"/> so initial spawn can match the rect.
    /// </summary>
    [DefaultExecutionOrder(-450)]
    public sealed class ParticleSimulationBounds : MonoBehaviour
    {
        [Tooltip("Rect that defines the simulation region in world space (GetWorldCorners).")]
        [SerializeField]
        private RectTransform _areaRect;

        [Tooltip("Shrinks the clamp box inward so particle centers stay away from the visual edge.")]
        [SerializeField]
        private float _margin;

        [Tooltip("If true, pushes bounds to ECS every Update (for moving or camera-driven UI).")]
        [SerializeField]
        private bool _syncEachFrame = true;

        public float Margin => math.max(0f, _margin);

        /// <summary>
        /// World-space axis-aligned bounds on the XY plane from the rect's world corners.
        /// </summary>
        public bool TryGetWorldAabb(out float2 min, out float2 max)
        {
            return _areaRect.TryGetWorldAabbXY(Margin, out min, out max);
        }

        private void Awake()
        {
            PushToWorldIfPossible();
        }

        private void Update()
        {
            if (_syncEachFrame)
                PushToWorldIfPossible();
        }

        private void PushToWorldIfPossible()
        {
            if (_areaRect == null)
                return;

            if (!TryGetWorldAabb(out var min, out var max))
                return;

            var world = World.DefaultGameObjectInjectionWorld;
            if (world == null || !world.IsCreated)
                return;

            var em = world.EntityManager;
            using var q = em.CreateEntityQuery(
                ComponentType.ReadOnly<SpatialGridMapTag>(),
                ComponentType.ReadWrite<SimulationWorldBounds>());
            if (q.IsEmptyIgnoreFilter)
                return;

            var e = q.GetSingletonEntity();
            em.SetComponentData(e, new SimulationWorldBounds
            {
                BoundsEnabled = 1,
                Min = min,
                Max = max,
                Margin = Margin
            });
        }
    }

    /// <summary>
    /// Non-Burst helpers for converting UI rects to simulation coordinates.
    /// </summary>
    public static class RectTransformSimulationBoundsUtility
    {
        public static bool TryGetWorldAabbXY(this RectTransform rt, float margin, out float2 min, out float2 max)
        {
            if (rt == null)
            {
                min = float2.zero;
                max = float2.zero;
                return false;
            }

            var corners = new Vector3[4];
            rt.GetWorldCorners(corners);
            min = new float2(float.PositiveInfinity, float.PositiveInfinity);
            max = new float2(float.NegativeInfinity, float.NegativeInfinity);
            for (var i = 0; i < 4; i++)
            {
                var xy = new float2(corners[i].x, corners[i].y);
                min = math.min(min, xy);
                max = math.max(max, xy);
            }

            //float doubleMargin = margin * 2;
            //min += new float2(doubleMargin, doubleMargin);
            //max -= new float2(doubleMargin, doubleMargin);

            return max.x > min.x && max.y > min.y;
        }
    }
}
