// Metaball Composite Pass
// Reads the accumulated splat texture and composites fluid onto the scene.
//
// Paint / oil-paint mode (default):
//   - Hard threshold: fluid is either fully opaque or fully transparent
//   - Flat color: no brightness variation across the fluid surface
//   - Nearest-type snapping: each pixel gets the exact color of its dominant fluid type
//   - Subtle darkened edge for depth, no neon glow

Shader "FluidSim/MetaballComposite"
{
    Properties
    {
        _MainTex ("Scene Texture", 2D) = "white" {}
        _SplatTex ("Splat Texture", 2D) = "black" {}
        _Threshold ("Fluid Threshold", Range(0.01, 2.0)) = 0.4
        _EdgeSoftness ("Edge Softness", Range(0.01, 0.5)) = 0.05
        _EdgeHighlight ("Edge Highlight", Range(-1.0, 1.0)) = -0.15
        _ColorSaturation ("Color Saturation", Range(0.5, 2.0)) = 1.1
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
            sampler2D _SplatTex;

            float _Threshold;
            float _EdgeSoftness;
            float _EdgeHighlight;
            float _ColorSaturation;
            float _SolidColors;
            float _FluidTypeCount;

            // Fluid type colors for nearest-color snapping (up to 8 types)
            float4 _FluidTypeColors[8];

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

            float3 AdjustSaturation(float3 color, float saturation)
            {
                float grey = dot(color, float3(0.299, 0.587, 0.114));
                return lerp(float3(grey, grey, grey), color, saturation);
            }

            /// Finds the fluid type color closest to the blended color.
            float3 SnapToNearestType(float3 blendedColor)
            {
                float bestDist = 99999.0;
                float3 bestColor = blendedColor;
                int count = (int)_FluidTypeCount;

                for (int i = 0; i < count; i++)
                {
                    float3 typeCol = _FluidTypeColors[i].rgb;
                    float3 diff = blendedColor - typeCol;
                    float dist = dot(diff, diff);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestColor = typeCol;
                    }
                }

                return bestColor;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float4 scene = tex2D(_MainTex, i.uv);
                float4 splat = tex2D(_SplatTex, i.uv);

                float totalWeight = splat.a;

                // No fluid here
                if (totalWeight < 0.001)
                    return scene;

                // Recover the average fluid color (weighted mean)
                float3 fluidColor = splat.rgb / totalWeight;

                // Snap to nearest fluid type color for clean flat colors
                if (_SolidColors > 0.5)
                {
                    fluidColor = SnapToNearestType(fluidColor);
                }

                // Adjust saturation
                fluidColor = AdjustSaturation(fluidColor, _ColorSaturation);

                // Edge detection: how close are we to the threshold boundary
                float edge = smoothstep(
                    _Threshold - _EdgeSoftness,
                    _Threshold + _EdgeSoftness,
                    totalWeight
                );

                // Edge shading: darken edges (negative _EdgeHighlight)
                // or brighten them (positive). For paint look, slight darkening
                // gives a nice depth illusion without neon glow.
                float edgeBand = smoothstep(
                    _Threshold - _EdgeSoftness * 3.0,
                    _Threshold,
                    totalWeight
                ) * (1.0 - smoothstep(
                    _Threshold,
                    _Threshold + _EdgeSoftness * 3.0,
                    totalWeight
                ));
                fluidColor += edgeBand * _EdgeHighlight;

                // Clamp to avoid any over-bright pixels
                fluidColor = saturate(fluidColor);

                // Composite: fully opaque fluid where edge = 1
                return lerp(scene, float4(fluidColor, 1.0), edge);
            }
            ENDCG
        }
    }
}
