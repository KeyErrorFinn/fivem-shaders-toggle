#include"ReShadeUI.fxh"
#include"ReShade.fxh"
#define NVE_BLOOM
#include"NVE/nve.fxh"
void PostProcessBloomVS(uint f:SV_VertexID,out float4 r:SV_Position,out float2 z:TEXCOORD,out float4 R:TEXCOORD1,out float4 E:TEXCOORD2){z.x=f==2?2.:0.;z.y=f==1?2.:0.;r=float4(z*float2(2.,-2.)+float2(-1.,1.),0.,1.);R=float4(EOctaveWeight1,EOctaveWeight2,EOctaveWeight3,EOctaveWeight4);E=float4(EOctaveWeight5,EOctaveWeight6,0.f,0.f);if(EFalloffType==1)R=float4(.001,.11,.25,.44),E=float4(.7,1.,0.,0.);if(EFalloffType==2)R=float4(.01,.2,.6,1.),E=float4(.6,.1,0.,0.);if(EFalloffType==3)R=float4(0.,.01,.2,.6),E=float4(.2,.01,0.,0.);if(EFalloffType==4)R=float4(.05,.8,.4,.1),E=float4(.05,.01,0.,0.);if(EFalloffType==5)R=float4(0.,.01,-.3,.7),E=float4(.2,.1,0.,0.);float m=1.-EFalloff;m*=m;R=lerp(R,1.,m);E=lerp(E,1.,m);}float4 texBiquadratic(sampler2D f,float2 d,float2 n,float2 m){float2 z=frac(m*d),E=(z*(z-1.)+.5)*n,R=m-E,w=m+E;float4 P=tex2D(f,float2(R.x,R.y))+tex2D(f,float2(R.x,w.y))+tex2D(f,float2(w.x,w.y))+tex2D(f,float2(w.x,R.y));return P*.25;}float3 FuncBlur(sampler2D f,float2 n,float m,float P){float2 z=4./m,R,E,w;float y=BUFFER_WIDTH/BUFFER_HEIGHT;z.y*=1/y;R=m;R*=z;R=min(R,16.);R=max(R,2.);int x=(int)(R.x+.4999),e=(int)(R.y+.4999);R=1./R;float4 r=0.;r.w=1e-6;w=.5*R.xy;E.x=-.5+w.x;z*=2.;for(int i=0;i<x;i++){E.y=-.5+w.y;for(int B=0;B<e;B++){float2 o=E.xy*z+n.xy,s;float3 t=tex2D(f,o.xy).xyz;float F,d;s=E.xy*2.;d=dot(s.xy,s.xy);F=saturate(1001.-1e3*d);F*=saturate(1.-d);r.xyz+=t.xyz*F;r.w+=F;E.y+=R.y;}E.x+=R.x;}r.xyz*=1./r.w;return r.xyz;}float4 PS_Threshold(float4 f:SV_Position,float2 E:TexCoord):SV_Target{float4 R;R.xyz=tex2D(ReShade::BackBuffer,E).xyz;R=max(R,0.);R=min(R,32768.);R*=smoothstep(0.f,1.f,pow(length(R)*threshold,thresholdcurve));return R;}float4 PS_Resize1024x512(float4 f:SV_Position,float2 E:TexCoord):SV_Target{float4 R;R.xyz=FuncBlur(TDSampler,E,1024.f,512.f);R=max(R,0.);R=min(R,32768.);R.w=1.;return R;}float4 PS_Resize512x256(float4 f:SV_Position,float2 E:TexCoord):SV_Target{float4 R;R.xyz=FuncBlur(RTSampler512,E,512.f,256.f);R=max(R,0.);R=min(R,32768.);R.w=1.;return R;}float4 PS_Resize256x128(float4 f:SV_Position,float2 E:TexCoord):SV_Target{float4 R;R.xyz=FuncBlur(RTSampler256,E,256.f,128.f);R=max(R,0.);R=min(R,32768.);R.w=1.;return R;}float4 PS_Resize128x64(float4 f:SV_Position,float2 E:TexCoord):SV_Target{float4 R;R.xyz=FuncBlur(RTSampler128,E,128.f,64.f);R=max(R,0.);R=min(R,32768.);R.w=1.;return R;}float4 PS_Resize64x32(float4 f:SV_Position,float2 E:TexCoord):SV_Target{float4 R;R.xyz=FuncBlur(RTSampler64,E,64.f,32.f);R=max(R,0.);R=min(R,32768.);R.w=1.;return R;}float4 PS_Resize32x16(float4 f:SV_Position,float2 E:TexCoord):SV_Target{float4 R;R.xyz=FuncBlur(RTSampler32,E,32.f,16.f);R=max(R,0.);R=min(R,32768.);R.w=1.;return R;}float4 BloomPass(float4 f:SV_Position,float2 E:TexCoord,float4 R:TEXCOORD1,float4 n:TEXCOORD2):SV_Target{float4 z;float r;float3 m,w,i,y,x,e,o=tex2D(ReShade::BackBuffer,E).xyz;m=tex2D(RTSampler512,E).xyz*R.x;m*=Color0;w=texBiquadratic(RTSampler256,256.,1./256.,E).xyz*R.y;w*=Color1;i=texBiquadratic(RTSampler128,128.,1./128.,E).xyz*R.z;i*=Color2;y=texBiquadratic(RTSampler64,64.,1./64.,E).xyz*R.w;y*=Color3;x=texBiquadratic(RTSampler32,32.,1./32.,E).xyz*n.x;x*=Color4;e=texBiquadratic(RTSampler16,16.,.0625,E).xyz*n.y;e*=Color5;z.xyz=m+w+i+y+x+e;r=dot(R,1.)+dot(n,1.);z.xyz*=1./r;z.xyz*=BloomIntensity;z.xyz*=1-ScanlineFilterParams.y;z.xyz*=max(1-DaylightTime,MinAmount);z=max(z,0.);z=min(z,32768.);z.w=1.f;return float4(o+z.xyz,1.f);}
technique NveBloom < ui_label = "NaturalVision Evolved: Bloom"; >
{   
    pass p0
    {
        VertexShader=PostProcessBloomVS;
        PixelShader=PS_Threshold;
        RenderTarget = TextureDownsampled;
    }

    pass p1
    {
        VertexShader=PostProcessBloomVS;
        PixelShader=PS_Resize1024x512;
        RenderTarget = RenderTarget512;
    }

    pass p2
    {
        VertexShader=PostProcessBloomVS;
        PixelShader=PS_Resize512x256;
        RenderTarget = RenderTarget256;
    }

    pass p3
    {
        VertexShader=PostProcessBloomVS;
        PixelShader=PS_Resize256x128;
        RenderTarget = RenderTarget128;
    }

    pass p4
    {
        VertexShader=PostProcessBloomVS;
        PixelShader=PS_Resize128x64;
        RenderTarget = RenderTarget64;
    }

    pass p5
    {
        VertexShader=PostProcessBloomVS;
        PixelShader=PS_Resize64x32;
        RenderTarget = RenderTarget32;
    }

    pass p6
    {
        VertexShader=PostProcessBloomVS;
        PixelShader=PS_Resize32x16;
        RenderTarget = RenderTarget16;
    }

    pass p7
    {
        VertexShader=PostProcessBloomVS;
        PixelShader=BloomPass;
    }
}
