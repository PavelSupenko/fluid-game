// Fluid Particle Splat — Alpha Blended
// Renders each particle as a colored circle with soft edges.
// Uses standard alpha blending (SrcAlpha, OneMinusSrcAlpha).
// Overlapping same-color particles merge seamlessly.
// Result: opaque colored blobs on transparent background.

Shader "FluidSim/MetaballSplat"
{
    Properties
    {
        _RenderScale ("Render Scale", Float) = 0.35
        _BlobSharpness ("Blob Sharpness", Float) = 2.0
    }

    SubShader
    {
        Tags { "Queue" = "Transparent" }

        // Standard alpha blending — NOT additive
        Blend SrcAlpha OneMinusSrcAlpha
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
                float density;  // Stores mass for merged particles
                float pressure;
                float alive;
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

                // Scale merged particles larger (area-preserving)
                float massScale = (p.density > 0.01) ? pow(p.density, 0.35) : 1.0;
                float scale = (p.alive > 0.5) ? _RenderScale * massScale : 0.0;

                float3 worldPos = float3(p.position, 0.0) + v.vertex.xyz * scale;
                o.pos = mul(_ViewProj, float4(worldPos, 1.0));
                o.uv = v.uv;
                o.color = p.color;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float2 offset = i.uv - 0.5;
                float dist = length(offset) * 2.0; // 0 at center, 1 at edge

                // Alpha: fully opaque at center, fades at edge
                float alpha = saturate(1.0 - pow(dist, _BlobSharpness));

                if (alpha < 0.01) discard;

                // Output: particle color with alpha for soft edges
                // Where circles overlap (same color), alpha blending
                // just reinforces the same color = seamless blob.
                return float4(i.color.rgb, alpha);
            }
            ENDCG
        }
    }
}
