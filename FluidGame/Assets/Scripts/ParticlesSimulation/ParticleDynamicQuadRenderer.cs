using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Serialization;

namespace ParticlesSimulation
{
    /// <summary>
    /// Draws all simulation particles as batched quads (single mesh, vertex colors, one shared texture slot).
    /// Intended as a lightweight placeholder until a GPU fluid surface replaces it.
    /// </summary>
    [DefaultExecutionOrder(1000)]
    public sealed class ParticleDynamicQuadRenderer : MonoBehaviour
    {
        [FormerlySerializedAs("material")] [SerializeField]
        private Material _material;
        [FormerlySerializedAs("albedo")] [SerializeField]
        private Texture2D _albedo;
        [FormerlySerializedAs("quadHalfExtent")] [SerializeField]
        private float _quadHalfExtent = 0.035f;
        [FormerlySerializedAs("renderLayer")] [SerializeField]
        private int _renderLayer;

        private Mesh mesh;
        private World world;
        private EntityQuery query;
        private bool initialized;
        private int capacity;

        private Vector3[] vertices;
        private Vector2[] uvs;
        private Color32[] colors;
        private int[] indices;

        public void Initialize(World world)
        {
            this.world = world;
            if (world == null || !world.IsCreated)
                return;

            var em = world.EntityManager;
            query = em.CreateEntityQuery(
                ComponentType.ReadOnly<ParticleCore>(),
                ComponentType.ReadOnly<ParticleDrawColor>(),
                ComponentType.ReadOnly<ParticleSimTag>());

            mesh = new Mesh { name = "ParticleFluidBatch", indexFormat = IndexFormat.UInt32 };
            EnsureCapacity(query.CalculateEntityCount());
            initialized = true;
        }

        private void OnDestroy()
        {
            if (mesh != null)
                Destroy(mesh);
        }

        private void LateUpdate()
        {
            if (!initialized)
                return;

            if (world == null || !world.IsCreated || _material == null || mesh == null)
            {
                if (mesh != null)
                    mesh.Clear();
                return;
            }

            if (query.IsEmptyIgnoreFilter)
            {
                mesh.Clear();
                return;
            }

            var count = query.CalculateEntityCount();
            if (count == 0)
            {
                mesh.Clear();
                return;
            }

            EnsureCapacity(count);

            using var cores = query.ToComponentDataArray<ParticleCore>(Allocator.TempJob);
            using var cols = query.ToComponentDataArray<ParticleDrawColor>(Allocator.TempJob);

            var h = _quadHalfExtent;
            for (var i = 0; i < count; i++)
            {
                var p = new float3(cores[i].position.x, cores[i].position.y, 0f);
                var f = cols[i].value;
                var c = new Color32(
                    (byte)(math.saturate(f.x) * 255f),
                    (byte)(math.saturate(f.y) * 255f),
                    (byte)(math.saturate(f.z) * 255f),
                    (byte)(math.saturate(f.w) * 255f));
                var v = i * 4;

                vertices[v + 0] = p + new float3(-h, -h, 0f);
                vertices[v + 1] = p + new float3(-h, h, 0f);
                vertices[v + 2] = p + new float3(h, h, 0f);
                vertices[v + 3] = p + new float3(h, -h, 0f);

                uvs[v + 0] = new Vector2(0f, 0f);
                uvs[v + 1] = new Vector2(0f, 1f);
                uvs[v + 2] = new Vector2(1f, 1f);
                uvs[v + 3] = new Vector2(1f, 0f);

                colors[v + 0] = c;
                colors[v + 1] = c;
                colors[v + 2] = c;
                colors[v + 3] = c;
            }

            mesh.SetVertices(vertices, 0, count * 4);
            mesh.SetUVs(0, uvs, 0, count * 4);
            mesh.SetColors(colors, 0, count * 4);
            mesh.SetTriangles(indices, 0, count * 6, 0);
            mesh.RecalculateBounds();

            if (_albedo != null)
                _material.SetTexture("_BaseMap", _albedo);

            var matrix = transform.localToWorldMatrix;
            Graphics.DrawMesh(mesh, matrix, _material, _renderLayer);
        }

        private void EnsureCapacity(int requiredParticles)
        {
            var neededVerts = math.max(4, requiredParticles * 4);
            if (capacity >= requiredParticles && vertices != null && vertices.Length >= neededVerts)
                return;

            capacity = math.max(requiredParticles, 16);
            var vc = capacity * 4;
            var tc = capacity * 2;
            vertices = new Vector3[vc];
            uvs = new Vector2[vc];
            colors = new Color32[vc];
            indices = new int[tc * 3];

            for (var i = 0; i < capacity; i++)
            {
                var v = i * 4;
                var t = i * 6;
                indices[t + 0] = v + 0;
                indices[t + 1] = v + 1;
                indices[t + 2] = v + 2;
                indices[t + 3] = v + 0;
                indices[t + 4] = v + 2;
                indices[t + 5] = v + 3;
            }
        }
    }
}
