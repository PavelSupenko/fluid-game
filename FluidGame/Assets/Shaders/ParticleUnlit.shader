Shader "FluidSim/ParticleUnlit"
{
    Properties
    {
        _BaseMap("Texture", 2D) = "white" {}

        // Note: We use _BaseColor instead of _ParticleColor to automatically 
        // sync with the URPMaterialPropertyBaseColor component from your C# code!
        _BaseColor("Base Color", Color) = (1, 1, 1, 1)
    }

    SubShader
    {
        Tags
        {
            "RenderType"="Transparent"
            "RenderPipeline"="UniversalPipeline"
            "Queue"="Transparent"
        }
        LOD 100

        // Optional: Standard Alpha Blending setup for particles
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off

        Pass
        {
            Name "Unlit"
            Tags
            {
                "LightMode"="UniversalForward"
            }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            // 1. CRITICAL: Enables DOTS Instancing and SRP Batcher compatibility
            #pragma multi_compile_instancing
            #pragma target 4.5

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID // Required for DOTS instancing
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID // Required to pass instance ID to fragment
            };

            // Texture definitions go OUTSIDE the CBuffer
            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            // 2. THE FIX: The strict UnityPerMaterial CBuffer for the SRP Batcher
            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                float4 _BaseColor;
            CBUFFER_END

            // 3. DOTS INSTANCING: This tells the GPU to look for per-entity data
            #ifdef UNITY_DOTS_INSTANCING_ENABLED
                UNITY_DOTS_INSTANCING_START(MaterialPropertyMetadata)
                    UNITY_DOTS_INSTANCED_PROP(float4, _BaseColor)
                UNITY_DOTS_INSTANCING_END(MaterialPropertyMetadata)
                
                #define _BaseColor UNITY_ACCESS_DOTS_INSTANCED_PROP_FROM_MACRO(float4, _BaseColor)
            #endif

            Varyings vert(Attributes input)
            {
                Varyings output = (Varyings)0;

                // Setup instance ID in vertex stage
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);

                output.positionHCS = TransformObjectToHClip(input.positionOS.xyz);
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                // Setup instance ID in fragment stage
                UNITY_SETUP_INSTANCE_ID(input);

                // Sample the texture
                half4 texColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv);

                // Get our per-entity color (Injected from the C# URPMaterialPropertyBaseColor)
                float4 instanceColor = _BaseColor;

                // Multiply texture by the instanced color
                return texColor * instanceColor;
            }
            ENDHLSL
        }
    }
}