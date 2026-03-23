Shader "FluidSim/ParticleCircleGPU"
{
    Properties
    {
        _RenderScale ("Render Scale", Float) = 0.12
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Transparent"
            "Queue" = "Transparent"
        }

        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5

            #include "UnityCG.cginc"

            // Must match FluidParticle / Particle struct layout (48 bytes)
            struct Particle
            {
                float2 position;    // 8
                float2 velocity;    // 8
                int typeIndex;      // 4
                float density;      // 4
                float pressure;     // 4
                float alive;        // 4
                float4 color;       // 16
            };

            // Particle data from the compute shader
            StructuredBuffer<Particle> _Particles;
            float _RenderScale;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv     : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos  : SV_POSITION;
                float2 uv   : TEXCOORD0;
                float4 color : COLOR0;
            };

            v2f vert(appdata v, uint instanceID : SV_InstanceID)
            {
                v2f o;

                Particle p = _Particles[instanceID];

                // Collapse dead particles to zero size
                float scale = (p.alive > 0.5) ? _RenderScale : 0.0;
                float3 worldPos = float3(p.position, 0.0) + v.vertex.xyz * scale;

                o.pos = mul(UNITY_MATRIX_VP, float4(worldPos, 1.0));
                o.uv = v.uv;
                o.color = p.color;

                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                // Distance from center of the quad for circular shape
                float2 offset = i.uv - 0.5;
                float dist = length(offset);

                // Smooth circle edge
                float alpha = 1.0 - smoothstep(0.35, 0.5, dist);

                float4 col = i.color;
                col.a *= alpha;

                // Discard fully transparent pixels
                clip(col.a - 0.01);

                return col;
            }
            ENDCG
        }
    }
}
