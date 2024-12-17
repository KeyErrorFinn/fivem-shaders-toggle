#include"ReShade.fxh"
#define NVE_RAIN
#include"NVE/nve.fxh"
#define S(a,b,t)smoothstep(a,b,t)
float N(float f){return frac(sin(f*12345.564f)*7658.76f);}float4 N14(float f){return frac(sin(f*float4(123.f,1024.f,1456.f,264.f))*float4(6547.f,345.f,8799.f,1564.f));}float3 N31(float f){float3 r=frac(float3(f,f,f)*float3(.1031f,.11369f,.13787f));r+=dot(r,r.yzx+19.19f);return frac(float3((r.x+r.y)*r.z,(r.x+r.z)*r.y,(r.y+r.z)*r.x));}float SawTooth(float f){return cos(f+cos(f))+sin(2.f*f)*.2f+sin(4.f*f)*.02f;}float DeltaSawTooth(float f){return.4f*cos(2.f*f)+.08f*cos(4.f*f)-(1.f-sin(f))*sin(f+cos(f));}float2 GetDrops(float2 f,float a,float r,float x){f*=a;float y=timer*x,t,z,s,c;float2 N=float2(0.f,0.f),e,d;f.y+=timer*(_RainGravity*1e-4f+x*.04);f*=float2(dropHorizontalScale,2.5f)*2.f;e=floor(f);float3 l=N31(e.x+(e.y+r)*546.3524f);d=frac(f);d-=.5;d.y*=4.;d.x+=(l.x-.5)*.5;y+=l.z*6.28;t=SawTooth(y);d.y+=t*2.;z=d.x*d.x;z*=DeltaSawTooth(y);d.y+=z;s=length(d);c=S(clamp(.1f*a,.03f,.4f)*_RainDropsSize,0.f,s);N=d*c;return N;}float2 random(float f,float r){float2 d=float2(23.1406926327793f,2.66514414269023f);return abs(frac(float2(cos(f*d.x),sin(r*d.y+166.6))));}float2 generateRandomShift(float f,float r,float a,float2 d){return(2.*random(ceil(a/6.2831853)+f,ceil(a/6.2831853)+r)-1.)*d;}float sinWave01(float f){return.5+.5*sin(f-1.5707963);}float3 getSprite(float2 f,float r){float x=pow(.32f+tex2D(dropletSampler,f*(1.-1./(float2(64.f,64.f)*r/BUFFER_SCREEN_SIZE))).w,8.f);return float3(tex2D(dropletnSampler,f*(1.-1./(float2(64.f,64.f)*r/BUFFER_SCREEN_SIZE))).xy.xy*tex2D(dropletSampler,f*(1.-1./(float2(64.f,64.f)*r/BUFFER_SCREEN_SIZE))).w*dropDisplacement,tex2D(dropletSampler,f*(1.-1./(float2(64.f,64.f)*r/BUFFER_SCREEN_SIZE))).w+x);}float4 FastBlur(float2 f,sampler2D r,float a){float2 d=a/BUFFER_SCREEN_SIZE;float4 y=tex2D(r,f);for(float x=0.;x<6.2831853;x+=.39269908125)for(float N=1./3.;N<=1.;N+=1./3.)y+=tex2D(r,f+float2(cos(x),sin(x))*d*N);y/=33.;return y;}float4 RainLensPass(float4 f:SV_Position,float2 a:TexCoord):SV_Target{if(RainValue<=0)discard;float r=BUFFER_WIDTH/BUFFER_HEIGHT,d;float2 N=a,x=float2(0.f,0.f),s=a,t;s*=-1;s.x*=1.7f;d=0.f;float3 y=float3(0.f,0.f,0.f);for(int c=0;c<64;++c){float E=3.5*float(c),e=5.*(timer*1e-4f+E)*3.f,z;float2 l=float2(.5f,.5f)+generateRandomShift(2.*float(c),float(c+56),e,r)*.5f;float3 G=getSprite(N-l,dropRadius+random(l.x,l.y).x*dropRadiusRand);z=S(0.f,.2f,sinWave01(e))*.075f*S(safezoneIn,safezoneOut,length((a-float2(.5f,.5f))*float2(1.f,r*2.f)));d+=abs(G.z*z*visibilityBig);x+=G.xy*z;y+=abs(G.z*z*10.f)*LightColour*lightInt;}t=GetDrops(s,1.25f/_RainScale,1.f,_RainFallSpeed*.001f)*_RainGlobalIntensity;d+=abs(t.x)*visibilitySmall;x+=t;t=GetDrops(s,1.5f/_RainScale,15.f,_RainFallSpeed*.001f)*_RainGlobalIntensity;d+=abs(t.x)*visibilitySmall;x+=t;t=GetDrops(s,2.5f/_RainScale,25.f,_RainFallSpeed*.001f)*_RainGlobalIntensity;d+=abs(t.x)*visibilitySmall;x+=t;float4 l=FastBlur(a+x,colorSampler2,bgBlur);l+=y*LightIntensity*.2f;return float4(l.xyz,saturate(abs(d*10.f)));}float atan2(float f,float r){if(r>0.)return atan(f/r);if(f>=0.&&r<0.)return atan(f/r)+3.1415926;if(f<0.&&r<0.)return atan(f/r)-3.1415926;if(f>0.&&r==0.)return 1.5707963;return f<0.&&r==0.?-1.5707963:f==0.&&r==0.?1.5707963:1.5707963;}float2 uv_polar(float2 f,float2 r){float2 d=f-r;float x=length(d),s=atan2(d.x,d.y);return float2(s,x);}float2 uv_lens_half_sphere(float2 f,float2 r,float d,float x){float2 a=uv_polar(f,r),t;float s=clamp(1.-a.y/d,0.,1.),c=sqrt(1.-pow(s-1.,2.)),l=atan2(1.-s,c),e=l-asin(sin(l)/x),z=1.-s-sin(e)*c/cos(e);t=r+float2(sin(a.x),cos(a.x))*z*d;return lerp(f,t,float(length(f-r)<d));}float4 RainLensFinalPass(float4 f:SV_Position,float2 r:TexCoord):SV_Target{if(RainValue<=0)discard;float2 a=r;float4 d=FastBlur(a,targetSampler,totalBlur),t=tex2D(colorSampler2,r);t.xyz=lerp(t.xyz,d.xyz,d.w*RainValue);return t;}
technique NveRainLens < ui_label = "NaturalVision Evolved: Screen Raindrops"; >
{
	pass p0
	{
		VertexShader = PostProcessVS;
		PixelShader = RainLensPass;
        RenderTarget = renderTarget;

        ClearRenderTargets = false;
	}
    pass p1
    {
        VertexShader = PostProcessVS;
		PixelShader = RainLensFinalPass;
    }
}
