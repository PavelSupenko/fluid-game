// Fluid Composite — Simple Alpha Overlay
// Reads the pre-rendered fluid texture (with alpha) and composites over the scene.
// Optionally snaps to palette for clean flat colors.

Shader "FluidSim/MetaballComposite"
{
    Properties
    {
        _MainTex ("Scene Texture", 2D) = "white" {}
        _FluidTex ("Fluid Texture", 2D) = "black" {}
        _SolidColors ("Solid Colors", Float) = 1.0
        _FluidTypeCount ("Fluid Type Count", Float) = 3.0
    }

    SubShader
    {
        ZTest Always
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _FluidTex;
            float _SolidColors;
            float _FluidTypeCount;
            float4 _FluidTypeColors[16];

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv     : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv  : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float3 SnapToNearestType(float3 color)
            {
                float bestDist = 99999.0;
                float3 bestColor = color;
                int count = min((int)_FluidTypeCount, 16);

                for (int i = 0; i < count; i++)
                {
                    float3 diff = color - _FluidTypeColors[i].rgb;
                    float dist = dot(diff, diff);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestColor = _FluidTypeColors[i].rgb;
                    }
                }
                return bestColor;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float4 scene = tex2D(_MainTex, i.uv);
                float4 fluid = tex2D(_FluidTex, i.uv);

                float alpha = fluid.a;

                // No fluid
                if (alpha < 0.01)
                    return scene;

                float3 fluidColor = fluid.rgb;

                // Snap to palette — exact original colors
                if (_SolidColors > 0.5)
                    fluidColor = SnapToNearestType(fluidColor);

                // Simple alpha composite: fluid over scene
                float3 result = lerp(scene.rgb, fluidColor, alpha);

                return float4(result, 1.0);
            }
            ENDCG
        }
    }
}
