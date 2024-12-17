#include "ReShadeUI.fxh"
#include "ReShade.fxh"

#include "NVE/nve.fxh"

uniform float VignetteDepth <
    ui_tooltip = "Depth at which the effect is applied.";
    ui_label = "Vignette: Depth";
	ui_type = "slider";
	ui_min = 0.0f;
	ui_max = 10.0f;
> = 4.0f;

uniform bool FirstPerson <hidden=true;> = false;


float4 VignettePass(float4 vpos : SV_Position, float2 tex : TexCoord) : SV_Target
{
	float4 color = tex2D(ReShade::BackBuffer, tex);
	float4 original = color;
	float distance = (1.0 + pow( (1.0f-tex.y*1.0f), 1.0f * 8.0f) * -1.0f); 
	float3 vigCol = float3(0.85,0.85,0.85)-VignettingColour.xyz;
	color.rgb -= vigCol*(1.0f-distance)*color.rgb;
	//color.rgb *= distance;
	//color.rgb = saturate(color.rgb);
	float4 final = lerp(original, color, smoothstep(0,1.0f,ReShade::GetLinearizedDepth(tex)*VignetteDepth+(FirstPerson*10))*ScanlineFilterParams.y);
	return final;
}

technique NveVignette < ui_label = "NaturalVision Evolved: Vignette"; >
{
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader = VignettePass;
	}
}