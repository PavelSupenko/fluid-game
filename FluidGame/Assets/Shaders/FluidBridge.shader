// Fluid Bridge Shader
// Draws tapered connections (trapezoids) between nearby same-type particles.
// Each instance reads a bridge struct: posA, posB, radiusA, radiusB, color.
// The quad is stretched between A and B, with width matching each particle's radius.

Shader "FluidSim/FluidBridge"
{
    SubShader
    {
        Tags { "Queue" = "Transparent" }

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

            struct BridgeData
            {
                float2 posA;
                float2 posB;
                float radiusA;
                float radiusB;
                float4 color;
            };

            StructuredBuffer<BridgeData> _Bridges;
            float4x4 _ViewProj;
            float _EdgeSoftness; // 0-1: how soft the edges of bridges are

            struct appdata
            {
                float4 vertex : POSITION; // x: 0 or 1 (A or B end), y: -0.5 or 0.5 (side)
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
                BridgeData b = _Bridges[instanceID];

                // Direction from A to B
                float2 dir = b.posB - b.posA;
                float len = length(dir);

                if (len < 0.0001)
                {
                    o.pos = float4(0, 0, 0, 0);
                    o.uv = float2(0, 0);
                    o.color = float4(0, 0, 0, 0);
                    return o;
                }

                float2 fwd = dir / len;
                float2 side = float2(-fwd.y, fwd.x); // Perpendicular

                // v.vertex.x: 0 = A end, 1 = B end
                // v.vertex.y: -0.5 to 0.5 = side offset
                float t = v.vertex.x; // 0..1 along bridge
                float sideOffset = v.vertex.y; // -0.5..0.5

                // Interpolate position along bridge
                float2 center = lerp(b.posA, b.posB, t);

                // Interpolate width (tapered: wider at bigger particle)
                float width = lerp(b.radiusA, b.radiusB, t);

                // Final world position
                float2 worldPos2D = center + side * sideOffset * width;
                float3 worldPos = float3(worldPos2D, 0.0);

                o.pos = mul(_ViewProj, float4(worldPos, 1.0));
                o.uv = float2(t, sideOffset + 0.5); // uv.y: 0..1 across width
                o.color = b.color;
                return o;
            }

            float _BridgeAlpha;

            float4 frag(v2f i) : SV_Target
            {
                // Soft edges on the sides of the bridge
                float sideDistFromCenter = abs(i.uv.y - 0.5) * 2.0; // 0 at center, 1 at edge
                float alpha = saturate(1.0 - pow(sideDistFromCenter, 2.0 / max(_EdgeSoftness, 0.01)));
                alpha *= _BridgeAlpha;

                if (alpha < 0.01) discard;

                return float4(i.color.rgb, alpha);
            }
            ENDCG
        }
    }
}
