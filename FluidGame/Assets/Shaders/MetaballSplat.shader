// Metaball Splat Pass
// Renders each particle as a soft disk with additive blending.
// The profile is flat in the center and drops off only near the edge,
// which produces uniform color density across overlapping particles
// (no bright centers, no neon glow).
//
// Output: RGB = particle_color * weight, A = weight
// The composite pass divides rgb/a to recover the average color.

Shader "FluidSim/MetaballSplat"
{
    Properties
    {
        _RenderScale ("Render Scale", Float) = 0.35
        _BlobSharpness ("Blob Sharpness", Float) = 3.0
    }

    SubShader
    {
        Tags { "Queue" = "Transparent" }

        // Additive blending: accumulate (color*weight) and (weight)
        Blend One One
        ZWrite Off
        ZTest Always
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5

            struct Particle
            {
                float2 position;
                float2 velocity;
                int typeIndex;
                float density;
                float pressure;
                float pad;
                float4 color;
            };

            StructuredBuffer<Particle> _Particles;
            float _RenderScale;
            float _BlobSharpness;
            float4x4 _ViewProj;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv     : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos   : SV_POSITION;
                float2 uv    : TEXCOORD0;
                float4 color : COLOR0;
            };

            v2f vert(appdata v, uint instanceID : SV_InstanceID)
            {
                v2f o;
                Particle p = _Particles[instanceID];
                float3 worldPos = float3(p.position, 0.0) + v.vertex.xyz * _RenderScale;
                o.pos = mul(_ViewProj, float4(worldPos, 1.0));
                o.uv = v.uv;
                o.color = p.color;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float2 offset = i.uv - 0.5;
                float dist = length(offset) * 2.0; // normalized: 0 at center, 1 at quad edge

                // Flat-top profile: weight is 1.0 in the center region,
                // then drops smoothly to 0 near the edge.
                // This avoids bright centers — all overlapping particles
                // contribute equally regardless of where you sample.
                float weight = 1.0 - smoothstep(0.5, 1.0, dist);

                if (weight < 0.005) discard;

                return float4(i.color.rgb * weight, weight);
            }
            ENDCG
        }
    }
}
