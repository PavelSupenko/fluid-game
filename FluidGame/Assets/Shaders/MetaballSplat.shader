// Metaball Splat Pass
// Renders each particle as a soft blob with additive blending.
// Output: RGB = particle_color * weight, A = weight
// When blobs overlap, they accumulate smoothly — this is what
// creates the "merging" effect when thresholded.

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

            // Must match FluidParticle struct layout (48 bytes)
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

                // Billboard quad positioned at particle location
                float3 worldPos = float3(p.position, 0.0) + v.vertex.xyz * _RenderScale;

                // Use the explicitly provided VP matrix (set by MetaballFluidRenderer)
                o.pos = mul(_ViewProj, float4(worldPos, 1.0));
                o.uv = v.uv;
                o.color = p.color;

                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                // Distance from center of quad (0 at center, 0.5 at edge)
                float2 offset = i.uv - 0.5;
                float distSqr = dot(offset, offset) * 4.0; // normalized to [0, 1] at edge

                // Smooth gaussian-like falloff
                // exp(-sharpness * dist²) gives a nice blob shape
                float weight = exp(-_BlobSharpness * distSqr);

                // Discard pixels with negligible contribution
                if (weight < 0.01) discard;

                // Output: accumulated (color * weight, weight)
                // The composite pass will divide rgb by a to get the average color
                return float4(i.color.rgb * weight, weight);
            }
            ENDCG
        }
    }
}
