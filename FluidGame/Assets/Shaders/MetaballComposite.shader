// Metaball Composite Pass
// Reads the accumulated splat texture (RGB = sum of color*weight, A = sum of weight)
// and composites a smooth fluid surface onto the scene.
//
// The threshold determines how much accumulated weight is needed to be considered "fluid".
// Edge softness controls anti-aliasing at the fluid boundary.
// A subtle highlight is added at the edges for a glossy fluid look.

Shader "FluidSim/MetaballComposite"
{
    Properties
    {
        _MainTex ("Scene Texture", 2D) = "white" {}
        _SplatTex ("Splat Texture", 2D) = "black" {}
        _Threshold ("Fluid Threshold", Range(0.01, 2.0)) = 0.4
        _EdgeSoftness ("Edge Softness", Range(0.01, 0.5)) = 0.08
        _EdgeHighlight ("Edge Highlight", Range(0.0, 1.0)) = 0.3
        _ColorSaturation ("Color Saturation", Range(0.5, 2.0)) = 1.2
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
            float4 _SplatTex_TexelSize; // (1/width, 1/height, width, height)

            float _Threshold;
            float _EdgeSoftness;
            float _EdgeHighlight;
            float _ColorSaturation;

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

            // Boost saturation for more vibrant fluid colors
            float3 AdjustSaturation(float3 color, float saturation)
            {
                float grey = dot(color, float3(0.299, 0.587, 0.114));
                return lerp(float3(grey, grey, grey), color, saturation);
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float4 scene = tex2D(_MainTex, i.uv);
                float4 splat = tex2D(_SplatTex, i.uv);

                float totalWeight = splat.a;

                // No fluid here — return scene unchanged
                if (totalWeight < 0.001)
                    return scene;

                // Recover the average fluid color (weighted mean)
                float3 fluidColor = splat.rgb / totalWeight;

                // Boost saturation for more vivid look
                fluidColor = AdjustSaturation(fluidColor, _ColorSaturation);

                // Smooth edge: transition from transparent to opaque
                float edge = smoothstep(
                    _Threshold - _EdgeSoftness,
                    _Threshold + _EdgeSoftness,
                    totalWeight
                );

                // Edge highlight: add a bright rim at the fluid boundary
                // This creates a subtle glossy/gel-like appearance
                float edgeGlow = smoothstep(
                    _Threshold - _EdgeSoftness * 2.0,
                    _Threshold,
                    totalWeight
                ) * (1.0 - smoothstep(
                    _Threshold,
                    _Threshold + _EdgeSoftness * 2.0,
                    totalWeight
                ));
                fluidColor += edgeGlow * _EdgeHighlight;

                // Composite fluid onto scene
                return lerp(scene, float4(fluidColor, 1.0), edge);
            }
            ENDCG
        }
    }
}
