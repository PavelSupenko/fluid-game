Shader "FluidSim/ParticleCircle"
{
    Properties
    {
        // This property is overridden per-instance via MaterialPropertyBlock.
        // Named _ParticleColor to avoid conflict with the built-in _Color property.
        _ParticleColor ("Particle Color", Color) = (1, 1, 1, 1)
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
            #pragma multi_compile_instancing

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv     : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv  : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            // Per-instance color property
            UNITY_INSTANCING_BUFFER_START(Props)
                UNITY_DEFINE_INSTANCED_PROP(float4, _ParticleColor)
            UNITY_INSTANCING_BUFFER_END(Props)

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);

                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);

                // Distance from center of the quad (UV 0.5, 0.5)
                float2 offset = i.uv - 0.5;
                float dist = length(offset);

                // Smooth circle edge with slight soft glow
                float alpha = 1.0 - smoothstep(0.35, 0.5, dist);

                float4 col = UNITY_ACCESS_INSTANCED_PROP(Props, _ParticleColor);
                col.a *= alpha;

                // Discard fully transparent pixels
                clip(col.a - 0.01);

                return col;
            }
            ENDCG
        }
    }
}
