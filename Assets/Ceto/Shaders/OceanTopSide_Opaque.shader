
Shader "Ceto/OceanTopSide_Opaque" 
{
	Properties 
	{
		[HideInInspector] _CullFace ("__cf", Float) = 2.0
	}
	SubShader 
	{
		Tags { "OceanMask"="Ceto_ProjectedGrid_Top" "RenderType"="Ceto_ProjectedGrid_Top" "IgnoreProjector"="True" "Queue"="AlphaTest+50" }
		LOD 200
		
		GrabPass { "Ceto_RefractionGrab" }
		
		zwrite on
		//cull back

		cull [_CullFace]
		
		CGPROGRAM
		#pragma surface OceanSurfTop OceanBRDF noforwardadd nolightmap
		//#pragma surface OceanSurfTop OceanBRDF nolightmap fullforwardshadows
		#pragma vertex OceanVert
		#pragma target 3.0
		
		#pragma multi_compile __ CETO_REFLECTION_ON
		#pragma multi_compile __ CETO_UNDERWATER_ON
		#pragma multi_compile __ CETO_USE_OCEAN_DEPTHS_BUFFER
		#pragma multi_compile __ CETO_USE_4_SPECTRUM_GRIDS
		#pragma multi_compile __ CETO_STERO_CAMERA
		
		//#define CETO_REFLECTION_ON
		//#define CETO_UNDERWATER_ON
		//#define CETO_USE_OCEAN_DEPTHS_BUFFER
		//#define CETO_USE_4_SPECTRUM_GRIDS
		
		//#define CETO_DISABLE_SPECTRUM_SLOPE
		//#define CETO_DISABLE_SPECTRUM_FOAM
		//#define CETO_DISABLE_NORMAL_OVERLAYS
		//#define CETO_DISABLE_FOAM_OVERLAYS
		//#define CETO_DISABLE_EDGE_FADE
		//#define CETO_DISABLE_FOAM_TEXTURE
		//#define CETO_DISABLE_CAUSTICS
		
		//#define CETO_BRDF_FRESNEL
		//#define CETO_NICE_BRDF
		#define CETO_OCEAN_TOPSIDE
		#define CETO_OPAQUE_QUEUE

		#include "./OceanShaderHeader.cginc"
		#include "./OceanDisplacement.cginc"
		#include "./OceanBRDF.cginc"
		#include "./OceanUnderWater.cginc"
		#include "./OceanSurfaceShaderBody.cginc"

		ENDCG
		
		Pass 
		{
			Name "ShadowCaster"
			Tags { "LightMode" = "ShadowCaster" }
			
			zwrite on 
			ztest lequal

			cull [_CullFace]
			
			CGPROGRAM
			#pragma vertex OceanVertShadow
			#pragma fragment OceanFragShadow
			#pragma target 3.0
		
			#pragma multi_compile_shadowcaster

			#pragma multi_compile __ CETO_USE_4_SPECTRUM_GRIDS
			#pragma multi_compile __ CETO_STERO_CAMERA
			//#define CETO_USE_4_SPECTRUM_GRIDS
	
			#include "UnityCG.cginc"
					
			#include "./OceanShaderHeader.cginc"
			#include "./OceanDisplacement.cginc"
			#include "./OceanShadowCasterBody.cginc"
			
			ENDCG	
		}
		
	} 
	
	FallBack Off
}















