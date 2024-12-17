#include "ReShadeUI.fxh"
#include "ReShade.fxh"
#include "NVE/nve.fxh"
#include "NVE/nve_clouds.fxh"

#define INV_SQRT_OF_2PI 0.39894228040143267793994605993439  // 1.0/SQRT_OF_2PI
#define INV_PI 0.31830988618379067153776752674503
#define Pi 3.141592f
#define TwoPi 6.28318530718
//#define RAYS
#define MOVEMENT

uniform float CloudyCoverage < hidden = true; > = 0.0f;

uniform float timer < source = "timer";>;

uniform float MyValue <	ui_type = "drag";ui_min = -1000.0f;ui_max = 1000.0f;hidden = true;> = 1.0f;
uniform float MyValue2 <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float MyValue3 <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float MyValue4 <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float MyValue5 <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
//uniform float3 myColor <	ui_type = "color";ui_min = 0.0f;ui_max = 1.0f;> = 1.0f;

uniform float earthShad1 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;hidden = true;> = 1.0f;
uniform float earthShad2 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;hidden = true;> = 1.0f;
uniform float earthShad3 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;hidden = true;> = 1.0f;
uniform float earthShad4 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;hidden = true;> = 1.0f;
uniform float earthShad5 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;hidden = true;> = 1.0f;
uniform float earthShad6 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;hidden = true;> = 1.0f;
uniform float earthShad7 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;hidden = true;> = 1.0f;
uniform float earthShad8 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;hidden = true;> = 1.0f;

texture2D           TextureNoise <source = "ns.png";>{Width = 64; Height = 64; Format = R8;}; 
texture3D           TextureNoise_actually3D <source = "3d.dds";>{Width = 128; Height = 128; Depth = 256; Format = R8;};          
texture3D           TextureNoise_actually3DCurl <source = "curl3D.dds";>{Width = 128; Height = 128; Depth = 256; Format = R8;};  

texture2D RenderTarget512_2     {	Width = 512; Height = 512; Format = RGBA32F;};
texture2D RenderTarget512_3     {	Width = 512; Height = 512; Format = RGBA32F;};
texture2D RenderTarget32        {	Width = 32; Height = 32; Format = RGBA32F;};
texture2D RenderTarget64        {	Width = 64; Height = 64; Format = RGBA32F;};
texture2D RenderTarget128       {	Width = 128; Height = 128; Format = RGBA32F;};
texture2D RenderTarget256       {	Width = 256; Height = 256; Format = RGBA32F;};
texture2D RenderTarget512       {	Width = 512; Height = 512; Format = RGBA32F;};
texture2D RenderTarget1024       {	Width = 1024; Height = 1024; Format = RGBA32F;};
texture2D RenderTarget1024_2     {	Width = 1024; Height = 1024; Format = RGBA32F;};
texture2D RenderTarget1024_3       {	Width = 1024; Height = 1024; Format = RGBA32F;};
texture2D RenderTargetHalf       {	Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA32F;};
texture2D RenderTargetFull      {	Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA32F;};
texture2D RenderTargetFull2      {	Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA32F;};
texture2D RenderTargetFullMask      {	Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8;};
texture2D RenderTargetFullMaskGrow      {	Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8;};
texture2D TextureDownsampled    {	Width = 1024; Height = 1024; Format = RGBA32F;};

sampler2D RTSampler512_2        { Texture = RenderTarget512_2; };
sampler2D RTSampler512_3        { Texture = RenderTarget512_3; };
sampler2D RTSampler32           { Texture = RenderTarget32; };
sampler2D RTSampler64           { Texture = RenderTarget64; };
sampler2D RTSampler128          { Texture = RenderTarget128; };
sampler2D RTSampler256          { Texture = RenderTarget256; };
sampler2D RTSampler512          { Texture = RenderTarget512; };
sampler2D RTSampler1024          { Texture = RenderTarget1024; };
sampler2D RTSampler1024_2       { Texture = RenderTarget1024_2; };
sampler2D RTSampler1024_3          { Texture = RenderTarget1024_3; };
sampler2D RTSamplerHalf          { Texture = RenderTargetHalf; };
sampler2D RTSamplerFull         { Texture = RenderTargetFull;};
sampler2D RTSamplerFull2         { Texture = RenderTargetFull2;};
sampler2D RTSamplerFullMask         { Texture = RenderTargetFullMask;};
sampler2D RTSamplerFullMaskGrow         { Texture = RenderTargetFullMaskGrow;};
sampler2D TDSampler             { Texture = TextureDownsampled; };

sampler2D Sampler2_ns
{
	Texture = TextureNoise;

	AddressU = WRAP;
	AddressV = WRAP;
	AddressW = WRAP;

	MagFilter = LINEAR;
	MinFilter = LINEAR;
	MipFilter = LINEAR;

	MinLOD = 0.0f;
	MaxLOD = 1000.0f;

	MipLODBias = 0.0f;

	SRGBTexture = false;
};

sampler3D Sampler3D
{
	Texture = TextureNoise_actually3D;

	AddressU = WRAP;
	AddressV = WRAP;
	AddressW = WRAP;

	MagFilter = LINEAR;
	MinFilter = LINEAR;
	MipFilter = LINEAR;

	MinLOD = 0.0f;
	MaxLOD = 0.0f;

	MipLODBias = 0.0f;

	SRGBTexture = false;
};

sampler3D Sampler3DCurl
{
	Texture = TextureNoise_actually3DCurl;

	AddressU = WRAP;
	AddressV = WRAP;
	AddressW = WRAP;

	MagFilter = LINEAR;
	MinFilter = LINEAR;
	MipFilter = LINEAR;

	MinLOD = 0.0f;
	MaxLOD = 0.0f;

	MipLODBias = 0.0f;

	SRGBTexture = false;
};


/* uniform float fUIFarPlane <
	ui_category = "Advanced settings"; 
	ui_type = "drag";
	ui_label = "Far Plane (Preview)";
	ui_tooltip = "RESHADE_DEPTH_LINEARIZATION_FAR_PLANE=<value>\n"
	             "Changing this value is not necessary in most cases.";
	ui_min = 0.0; ui_max = 60000.0;
	ui_step = 1.0;
> = RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;

uniform float fUIDepthMultiplier <
	ui_category = "Advanced settings"; 
	ui_type = "drag";
	ui_label = "Multiplier (Preview)";
	ui_tooltip = "RESHADE_DEPTH_MULTIPLIER=<value>";
	ui_min = 0.0; ui_max = 60000.0;
	ui_step = 0.1;
> = RESHADE_DEPTH_MULTIPLIER;

uniform int iUILogarithmic <
		ui_type = "combo";
		ui_label = "Logarithmic (Preview)";
		ui_items = "RESHADE_DEPTH_INPUT_IS_LOGARITHMIC=0\0"
		           "RESHADE_DEPTH_INPUT_IS_LOGARITHMIC=1\0";
		ui_tooltip = "Change this setting if the displayed surface normals have stripes in them.";
	> = RESHADE_DEPTH_INPUT_IS_LOGARITHMIC;

    uniform int iUIReversed <
		ui_type = "combo";
		ui_label = "Reversed (Preview)";
		ui_items = "RESHADE_DEPTH_INPUT_IS_REVERSED=0\0"
		           "RESHADE_DEPTH_INPUT_IS_REVERSED=1\0";
	> = RESHADE_DEPTH_INPUT_IS_REVERSED; */

float GetLinearizedDepth(float2 texcoord, float addMult)
{       
	float depth = tex2Dlod(ReShade::DepthBuffer, float4(texcoord, 0, 0)).x * 16.0;
	const float C = 0.01;

	depth = 1.0 - depth;
	const float N = 1.0;
	depth /= 2.0 - depth * (2.0);
	return depth;	
}
float GetLinearizedDepthForRays(float2 texcoord, float addMult)
{    
	float depth = tex2Dlod(ReShade::DepthBuffer, float4(texcoord, 0, 0)).x * 7.0f * addMult;
	const float C = 0.01;
	const float N = 1.0;
	depth /= 2.0 - depth * (1.0f - N);
	return depth;	
}
float CustomDepth(float2 texcoord){ //This is some crazy trickery to scale depth
    float depth = tex2Dlod(ReShade::DepthBuffer, float4(texcoord, 0, 0)).x * 16.0;

    depth /= 2.0 - depth * 2.0;
    //depth = depth <= 0.0 ? depth : depth;
    //depth = depth > fUIFarPlane ? 0.000001 : depth;

    float dynamicDepth = (1.0f-pow(abs(texcoord * 2.0 - 1.0),1.5)); //pseudo fix depth fov
    depth = min(1.0/depth,100000);

    return depth-(dynamicDepth*0.08)*depth;
}

//==================NVE CLOUDS====================
static const float3 _baseColor = float3(0.0f,0.0f,0.0f);
static const float3 _colorNight = float3(0.0f,0.0f,0.0f);
static const float3 _baseColorDay = float3(0.0f,0.0f,0.0f);
static const float3 _baseColorSunset = float3(0.0f,0.0f,0.0f);
static const float3 _sunColorSunRise = float3(0.0f,0.0f,0.0f);
static const float3 _sunColorSunDay = float3(0.0f,0.0f,0.0f);
static const float3 _sunColorSunSet = float3(0.0f,0.0f,0.0f);

//===================NOISEGEN=====================
static float randomSeed = 1618.03398875;

float hash(float n) {
    return frac(sin(n/1873.1873) * randomSeed);
}
float noise2d(float3 p) {

    return tex2Dlod(Sampler2_ns,float4(p.x,p.y,0.0f,0.0f)).x;
}
float noise3d(float3 p) {

    float3 fr = floor(p),
    ft = frac(p);

    float n = 1153.0 * fr.x + 2381.0 * fr.y + fr.z,
    nr = n + 1153.0,
    nd = n + 2381.0,
    no = nr + 2381.0,

    v = lerp(hash(n), hash(n + 1.0), ft.z),
    vr = lerp(hash(nr), hash(nr + 1.0), ft.z),
    vd = lerp(hash(nd), hash(nd + 1.0), ft.z),
    vo = lerp(hash(no), hash(no + 1.0), ft.z);

    return lerp(lerp(v,vr,ft.x), lerp(vd,vo,ft.x),ft.y);
}
float sampledNoise3D_3D(float3 p){;
    p*=0.1f;

    //p.z+=4096;

    float ret = 1.0f-tex3Dlod(Sampler3D,float4(p.x,p.y,p.z,0.0f)).x;
	
	return ret;
	
}
float sampledNoise3D_3DCurl(float3 p){;
    p*=0.1f;

    //p.z+=4096;

    float ret = 1.0f-tex3Dlod(Sampler3DCurl,float4(p.x,p.y,p.z,0.0f)).x;
	
	return ret;
	
}
float map(float x, float a, float b, float c, float d) {
    return (x - b) / (a - b) * (c - d) + d;
}
float clampMap(float x, float a, float b, float c, float d) {
    return saturate((x - b) / (a - b)) * (c - d) + d;
}

//=================Structs========================
struct CloudProfile
{
    float4 march;   //min length, max length, multiplicand, max step
    
    float2 cutoff;  // density cut, transparency cut

    float2 volumeBox;   //top, bottom
    float4 shape;       //top, mid, bot, thickness

    float brightness;
    float3 range;    //total, top, bottom
    float2 _solidness;   //top, bottom
    
    float2 densityChunk;    //dens A,B
    
    float4 shadow;  //step length, detail strength, expanding, strength
    float4 distortion;  //max angle, strength, bump strength, small bump strength
    
    float fade;
    
    float3 densityDetail;   //dens C,D,E
    
    // 
    float3 scaleChunk;    //scale A,B, vertical stretch
    float3 scaleDetail;   //scale C,D,E
    float3 cloudShift;

    float3 offsetA;
    float3 offsetB;
    float3 offsetC;
    float3 offsetD;
    float3 offsetE;
    float startDistance;

};

struct CloudBaseColor
{
    float3 BaseColor;
    float3 BaseColor_Day;
    float3 BaseColor_Sunset;
};
#define CLOUD_COVERAGE float(coverageBottom*CloudyCoverage)
//#define CLOUD_COVERAGE float(1.0f)

struct AtmoColorPerDayTime
{
    float3 MorningAtSun;
    float3 MorningAwaySun;
    float3 Day;
    float3 EveningAtSun;
    float3 EveningAwaySun;
};

//=================Cloud settings========================
static CloudProfile cloudProfile;
static CloudProfile cloudProfile2;		
static CloudBaseColor baseColor;

//=================Functions=======================
float3 PosOnPlane(const in float3 origin, const in float3 direction, const in float h, inout float _distance) {
    _distance = (h - origin.z) / direction.z;
    return origin + direction * _distance;
}
float4 CloudShape(float z, float4 shape, float3 range) {
    float soft = map(z, shape.y, shape.x, range.z, range.y)*softMul;

    return float4(
        abs( smoothstep(shape.z, lerp(shape.y, shape.z, shape.w), z) * smoothstep(shape.x, lerp(shape.y, shape.x, shape.w), z)), 
        abs( range.x + soft ),
        abs( range.x - soft ),
        abs(soft));
}
float3 Distortion(float lump, float4 distortion)
{
    return float3(cos(lump*distortion.x)*distortion.y, 0.0, -lump*distortion.z);
}
float Chunk(in float3 pos, const in float2 density, const in float3 scale, const in float3 cloud_shift, const in float3 offsetA, const in float3 offsetB, in float cs) {
 
    float3 pp = pos;
    pos.z /= scale.z;
    pos += cloud_shift * pos.z;
    float3 cpPos = pos;

    //cpPos *= 0.1f;
    cpPos.xy *= cloudmapScale*0.01f;
    cpPos.z *= cloudmapZScale*0.01f;

    float3 p2 = pos;
    p2.z *= cloudmapZScale*50;
   
    float3
        pA = (pos + offsetA)*scale.x,
        pB = (pos + offsetB)*scale.y;
        pB.z *= chunkBZScale*10.0f;
  
   float dens_a = sampledNoise3D_3DCurl(pA).x * (sampledNoise3D_3DCurl(pB).x*density.y + density.x) * cs;
    //float altitudeParam = smoothstep(0.1,0.8,((pp.z)-cloudProfile.volumeBox.y)/(cloudProfile.volumeBox.x - (cloudProfile.volumeBox.y+cloudmapOffset)));
    //dens_a -= lerp((sampledNoise3D_3D(cpPos)),0.0f,1.0f-cloudMapStrength);

    return dens_a;
}

//Detail noise for clouds
float DetailA(in float3 pos, const in float3 density, const in float3 scale, const in float3 offsetC, const in float3 distortion) {
    float3 p = pos;
    p.z *= detailZScaleA;
    return density.x * sampledNoise3D_3D((p + offsetC + distortion) * scale.x).x;
}

float DetailB(const in CloudProfile cp,const in float lump_density, in float3 pos, const in float2 volumeBox, const in float3 density, const in float3 scale, const in float4 distortionParm, const in float3 offsetC, const in float3 offsetD,const in float3 offsetE, float cs) {
    float3 d = Distortion(lump_density, distortionParm);
    
    float3
        pD = pos + offsetD;

    float dens = DetailA(pos, density, scale, offsetC, d);

    d.z -= dens * distortionParm.w;

    float3 pD2 = pD;
    pD2.z *= detailZScaleB*2;

    float3 pD3 = pos + offsetE;
    pD3.z *= detailZScaleC*2;

    float param = saturate(densExp-exp(-dens));
    float param2 = 1.0f-saturate(exp(-dens));
    float altitudeParam = smoothstep(0.1,0.8,((pos.z)-cp.volumeBox.y)/(cp.volumeBox.x - (cp.volumeBox.y+altitudeOffsetB)));

    dens += density.y * 5 * (sampledNoise3D_3D((pD2 + d/3.0) * scale.y))*altitudeParam*param;
    float curl = sampledNoise3D_3DCurl((pD3 + d*8.0) * scale.z)*smoothstep(C_Smooth,C_Smooth2,dens*C_Smooth3);
    float c_contrast = saturate(pow(abs(curl*2-1),1 / max(C_Contrast,0.0001)) * sign(curl - 0.5) + 0.5);
    dens += density.z * c_contrast;
    //dens = map(dens+c_contrast+density.y * 5 * (sampledNoise3D_3DCurl((pD2 + d/3.0) * scale.y)),1.0-density.z,1.0,0.0,1.0);
    return dens;
}
float DensityField(float lump, float detail)
{
    return (lump*detail+lump);
}
float GetDensity(float density_field, float height, float low, float high, float2 volumeBox, float2 _solidness)
{
    return clampMap(density_field, low, high, 0.0, clampMap(height, volumeBoxY, volumeBox.x, _solidness.y, _solidness.x));
}

//==============SCATTERING FUNCTIONS=========================
float3 pow3(float3 v, float n)
{
    return float3(pow(v.x, n), pow(v.y, n), pow(v.z, n));
}

#define atmosphereStep (15.0)
#define lightStep (3.0)
#define fix (0.00001)
#define sunLightStrength ((sunStrength*100.0)*float3(1.0, 0.96, 0.949))
#define moonReflecticity (0.125)
#define starStrength (0.25)
#define LightingDecay (300.0)
#define rayleighStrength (1.85)
#define rayleighDecay (800.0)
#define atmDensity (atmDensityConfig)
#define atmCondensitivity (1.0)
#define waveLengthFactor float3(1785.0, 850.0, 410.0)
#define scatteringFactor (float3(1.0, 1.0, 1.0)*waveLengthFactor/rayleighDecay)  // for tuning, should be 1.0 physically, but you know.
#define mieStrength (0.125 + 0.5*0.1)
#define mieDecay (100.0)
#define fogDensity (0.3)
#define earthRadius (6.670)
#define  groundHeight (6.480)
#define AtmOrigin float3(0.0, 0.0, groundHeight)
#define earth float4(0.0, 0.0, 0.0, earthRadius)
#define game2atm (atmoConvDist*1000000.0f)
static float3 sunColor;

float3 Game2Atm(float3 gamepos)
{
    return (gamepos/(atmoConvDist*1000000.0f)) + AtmOrigin;
}

float4 sphereCast(float3 origin, float3 ray, float4 sphere, float _step, out float3 begin)
{
    float3 p = origin - sphere.xyz;
    
    float 
        r = length(p),
        d = length(cross(p, ray));    
        
    if (d > sphere.w+fix) return 0.0;    
        
    float
        sr = sqrt(sphere.w*sphere.w - d*d),    
        dr = -dot(p, ray);    
        
    float3
        pc = origin + ray * dr,
        pf = pc + ray * sr,
        pb = pc - ray * sr;
    
    float sl; 
    
    if(r > sphere.w){      
        begin = pb;
        sl = sr*2.0/_step;
    }
    else{   
        begin = origin;
        sl = length(pf - origin)/_step;
    }
    return float4(ray * sl, sl);
}

float2 Density(float3 pos, float4 sphere, float strength, float condense)
{
    float r = groundHeight;
    float h = length(pos-sphere.xyz)-r;
    float ep = exp(-(sphere.w-r)*condense);

    float fog = fogDensity*(1.0/(1200.0*h+0.5)-0.04)/1.96;
    
    if(h<0.0)
        return float2(strength, fogDensity);

    return float2((exp(-h*condense)-ep)/(1.0-ep)*strength, fog);
}

float3 rayleighScattering(float c)
{
    return (1.0+c*c)*rayleighStrength/waveLengthFactor;
}

float MiePhase(float c)
{
    return 1.0f+0.5f*exp(50.0f*(c-1.0f));
}

float PhaseFunc(float mu, float inG){

	return 
				(1.-inG * inG)
	/ //-----------------------------------
	(pow(1.+inG*inG - 2.0f * inG*mu, 1.5f)*4.0f* Pi);

}

float MieScattering(float c)
{
    return mieStrength*MiePhase(c);
}

float3 LightDecay(float densityR, float densityM)
{
    return float3(exp(-densityR/scatteringFactor - densityM*mieDecay));
}

float3 SunLight(float3 light, float3 position, float3 lightDirection, float4 sphere)
{
    float3 smp;
    float4 sms = sphereCast(position, lightDirection, sphere, lightStep, smp);      
    float2 dl;
    
    for(float j=0.0; j<lightStep; j++)
    {
        smp+=sms.xyz/2.0;
        dl += Density(smp, sphere, atmDensity, atmCondensitivity)*sms.w;
        smp+=sms.xyz/2.0;
    }

    return light*LightDecay(dl.x, dl.y)/LightingDecay;
}

float3 LightSource(float3 SunDir,float3 MoonDir, out float3 SourceDir)
{
    //float val1 = earthShadVal1;
    //float val2 = earthShadVal2;
    //float val3 = earthShadVal3;
    //float val4 = earthShadVal4;
    //float val5 = earthShadVal5;
    //float val6 = earthShadVal6;
    //float val7 = earthShadVal7;

    float val1 = earthShad1;
    float val2 = earthShad2;
    float val3 = earthShad3;
    float val4 = earthShad4;
    float val5 = earthShad5;
    float val6 = earthShad6;
    float val7 = earthShad7;
    float val8 = earthShad8;
    
    //const float val1 = -0.035;
    //const float val2 = 0.0;
    //const float val3 = -0.035;
    //const float val4 = 0.0;
    //const float val5 = -0.35;
    //const float val6 = -0.035;
    //const float val7 = -0.1;
    //const float val8 = 0.0;
    if(SunDir.x < 0.5){ //SUNSET

        if(SunDir.z<val1)
        {   
            SourceDir = MoonDir;
            return sunLightStrength*moonReflecticity*smoothstep(val2, val3, SunDir.z);
        }
        SourceDir = SunDir;
        return sunLightStrength*smoothstep(val3, val4, SunDir.z)*float3(0.3,0.3,0.3);
    }else{ 
        if(SunDir.z<val5)
            {   
            SourceDir = MoonDir;
            return sunLightStrength*moonReflecticity*smoothstep(val6, val7, SunDir.z);
        }
        SourceDir = SunDir;
        return sunLightStrength*smoothstep(val7, val8, SunDir.z)*float3(0.3,0.3,0.3);
    }
}

float ShadowMarching(float dens, float3 p, float2 densityChunk, float3 densityDetail, float3 scaleChunk, float3 scaleDetail, float3 shift, float4 profile, float3 dens_thres, const in float3 offsetA, const in float3 offsetB, const in float3 offsetC, float3 SunDir, float4 shadow, float2 volumeBox, float2 _solidness, float4 distortionParm) {
    if(dens<=shadowEarlyExit*0.0001f) return dens*shadow.x*shadowEarlyExitApprox;  //return a approx. value
    
    const float threshold = shadowThreshold/shadow.w/shadow.x;
    
    float d = 0.0;
    float4 _step = float4(SunDir.xyz * shadow.x, shadow.x);
    _step.xyz += noise2d(p)*SunDir.xyz*shadowNoise;
    

    for(int i=0; d<threshold && p.z<volumeBox.x && p.z>volumeBox.y && i<shadowSteps; i++)
    {
        float4 cs = CloudShape(p.z, profile, dens_thres);
        float d1 = Chunk(p, densityChunk, scaleChunk, shift, offsetA, offsetB, cs.x);
        
        float3 displace = Distortion(d1, distortionParm);
        float d2 = DetailA(p,densityDetail, scaleDetail, offsetC, displace) * shadow.y;
        d += GetDensity(DensityField(d1, d2*shadow.y), p.z, cs.z-shadow.z+shadowOffset, cs.y, volumeBox, _solidness);
        p+=_step.xyz;
    }
    return d*shadow.w*_step.w;
}
float ShadowOnGround(CloudProfile a, float3 position, float3 SunDirection)
{    
    a.range.x = 1.0f/a.range.x;
        
	float d;
    //float3 samppos = PosOnPlane(position, SunDirection, a.shape.y-C_fadeDist*100, d);
	float3 samppos = PosOnPlane(position, SunDirection, a.shape.y, d);

    float d1 = 0.0, d2 = 0.0;
    float4 cs;
    
	cs = CloudShape(samppos.z, a.shape, a.range);
	d1 = Chunk(samppos, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, cs.x);
	
	float3 displace = Distortion(d1, a.distortion);
	d2 = DetailA(samppos, a.densityDetail, a.scaleDetail, a.offsetC, displace) * a.shadow.y;    
	float adjustTerm =  max(a.shape.y - max(a.volumeBox.y, position.z), 0.0) * clampMap(d, 0.0, a.fade*0.8f, 1.0, 0.0);
    
	return GetDensity(DensityField(d1, d2), samppos.z, cs.z-a.shadow.z*2.0f, cs.y*5.0f, a.volumeBox, a._solidness)*adjustTerm;
}
//SPHERE CAST LAYER
// WIP SPHERE CAST FUNCTIONS

#define cloudBallRadius cBallRad
#define cloudClipAltitudeBegin cBallClipBegin
#define cloudClipAltitudeEnd cBallClipEnd
#define penetrationFix -1.0
#define lowerboundFix -50.0

float4 interPointSphere(float3 origin, float3 ray, float4 sphere)
{
    float3 p = origin - sphere.xyz;

    float 
        r = length(p),
        d = length(cross(p, ray));   
        
    if (d > sphere.w+fix) return float4(0.0,0.0,0.0,0.0);    
        
    float
        sr = sqrt(sphere.w*sphere.w - d*d),    
        dr = -dot(p, ray)+penetrationFix,    
        tmp = max(dr-sr, 0.0);

    if(dr-sr >= -16.0)
        return float4(ray * tmp, tmp);
    else
        return float4(ray * (dr+sr), (dr+sr));
}

float3 CastStep(float3 origin, float3 ray, float2 volumeBox, out float d)
{
    float3 origin_ec = origin;
    float r = origin.z+cloudBallRadius;
    origin_ec.xy = float2(0.0,0.0);
    volumeBox.x += cloudBallRadius;
    volumeBox.y += cloudBallRadius;
    
    if(r<volumeBox.y){
        float4 inner = interPointSphere(origin_ec, ray, float4(0.0,0.0,-cloudBallRadius,volumeBox.y));
        d = inner.w;
        return inner.xyz+origin.xyz;
    }
    else if(r<volumeBox.x)
    {
        d = 0.0;
        return origin;
    }
    else{
        float4 outer = interPointSphere(origin_ec, ray, float4(0.0,0.0,-cloudBallRadius,volumeBox.x));
        d = outer.w;
        return outer.xyz+origin.xyz;
    }
}

float4 fakeSphereMap(float3 p, float3 camera, float3 ray)
{
    float3 fromCenter = p-float3(camera.x,camera.y,-cloudBallRadius);
    float z = length(fromCenter)-cloudBallRadius;
    return float4(p.x, p.y, z, dot(ray, fromCenter));
}

float3 AtmosphereScattering(float3 background ,float3 marchPos, float4 marchStep, float3 ray, float3 lightStrength, float3 lightDirection, float strength, float4 sphere)
{
    float3 intensity=0.0;
    
    float ang_cos = dot(ray, lightDirection);
    float mie = MieScattering(ang_cos);
    float3 raylei = rayleighScattering(ang_cos);
    
    if(marchStep.w>0.015)
        marchStep /= marchStep.w/0.015;
    
    float2 dv=0.0;
    for(float i=0.0; i<atmosphereStep; i++)
    {
        float3 smp;
        float4 sms = sphereCast(marchPos, lightDirection, sphere, lightStep, smp);
        float2 sampling = Density(marchPos, sphere, atmDensity, atmCondensitivity)*marchStep.w;
            
        dv += sampling/2.0;
        
        float2 dl=dv;
        for(float j=0.0; j<lightStep; j++)
        {
            smp+=sms.xyz;
            dl += Density(smp, sphere, atmDensity, atmCondensitivity)*sms.w;
        }
        
        intensity += LightDecay(dl.x, dl.y)*(raylei*sampling.x + mie*sampling.y);     
        dv += sampling/2.0;
        marchPos+=marchStep.xyz;
    }
    
    return lightStrength*intensity*strength + background*LightDecay(dv.x, dv.y);
}

float3 atmosphere_scattering(float strength, float3 color, float3 camera, float3 ray, float distance, float3 SunDirection, float4 sphere)
{
    if(distance<200.0)
        return color;
    float fade=smoothstep(200.0, 300.0, distance);

    float4 marchStep=0.0;
    marchStep.w = 15.0f*distance/atmosphereStep/(atmoConvDist*1000000.0f);
    marchStep.xyz = ray*marchStep.w;
    
    float3 lightDirection = 0.0;
    float3 lightStrength = LightSource(SunDirection,MoonDirection.xyz, lightDirection);
    
    float3 scattered = AtmosphereScattering(color,  float3(0.0,0.0,camera.z/(atmoConvDist*1000000.0f)) + AtmOrigin, marchStep, ray, lightStrength, lightDirection, 1.0, sphere);
    return lerp(color, scattered, fade);
}

float4 CloudAtRay(in float3 bg_col,CloudProfile a, CloudBaseColor b, const in float3 cam_dir, in float3 cam_pos, float3 light, float3 lightDirection, float Time, in out float _distance, float2 uv) {
    float4 fx = float4(0.0, 0.0, 1.0,1.0); 
    float suncross = dot(lightDirection, cam_dir);
	a.range.x = 1.0f/a.range.x;

    float4 d = float4(0.0, 0.0, 0.0, a.march.y); 
    float3 p = CastStep(cam_pos, cam_dir, float2(a.volumeBox.x,a.volumeBox.y), d.x);
    d.y = d.x;

    float3 p_;

    if (d.x >= 0.0 && _distance>d.x) {
        
        float atm_bleed = 0.0f;
        
        float last_sample = 0.0, pdf=0.0;

        float suncross = dot(lightDirection, cam_dir);

        float dens;
         
        for (int i = 0; fx.z > 0.0 && i < a.march.w && d.x - d.w < _distance && d.x < a.fade; i++) {
            float4 fmp = fakeSphereMap(p, cam_pos, cam_dir);
            p_ = fmp.xyz;
            
            if((p_.z > a.volumeBox.x && fmp.w>10) || p.z<cloudClipAltitudeEnd){
                break;
            }
            else if (p_.z < a.volumeBox.y && fmp.w<10){
                p = CastStep(p+cam_dir, cam_dir, float2(a.volumeBox.x,a.volumeBox.y), d.z);
                i=0;
                continue;
            }

            float3 
                cs = CloudShape(p_.z, a.shape, a.range).xyz;

            float
                d1 = Chunk(p_, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, cs.x),
                d2 = DetailB(a,d1, p_, a.volumeBox, float3(a.densityDetail.x,a.densityDetail.y,a.densityDetail.z), a.scaleDetail, a.distortion, a.offsetC, a.offsetD,a.offsetE, cs.x),
                df = DensityField(d1, d2);
          
            if(df>cs.z+dfLimit)
            {              
                dens = GetDensity(df, p_.z, cs.z, cs.y, a.volumeBox, a._solidness);
                float c_dens = (dens + last_sample) * (a.march.x/2.0f);
                    
                last_sample = dens;

                if (d.x >= _distance)
                    c_dens *= d.z / d.w;
                
                if(c_dens > 0.0f)
                    d.y = d.y * (1.0 - fx.z) + fx.z * d.x;
                    
                fx.y += c_dens;
                fx.w += c_dens*0.1;
                fx.z = (exp(-fx.y*(edgeSmooth*0.1f)) - a.cutoff.y) / (1.0f - a.cutoff.y);      
                d.z = _distance - d.x;

                float dens_light=0.0;
                if(fx.y<shadowLimit)
                {
                    dens_light = 
                    ShadowMarching(
                        c_dens ,p_, a.densityChunk, a.densityDetail,
                         a.scaleChunk, a.scaleDetail, a.cloudShift,
                          a.shape, a.range, a.offsetA, a.offsetB, a.offsetC,
                           lightDirection, float4(a.shadow.x,a.shadow.y * (1.0f-saturate((d.x / (a.fade*0.750f*6.0f) ))),a.shadow.z,a.shadow.w), a.volumeBox, a._solidness, a.distortion
                           ); 
                    
                    if(usePowder == 1){
                        fx.x += c_dens * exp((-dens_light - fx.y)) * lerp(1.0f,1.0f-(exp((-dens)*PowderExp*PowderExp)),saturate(PowderStrength*(1.0f-suncross)));
                    }else{
                        fx.x += c_dens * exp((-dens_light - fx.y));
                    }
                }
            }

            float mask1 = clamp(RMMask1-clamp(d.x-MaskDistOffset,0.0f,a.fade)/(a.fade),0.0f,2.0f);
            float mask2 = clamp(RMMask2+clamp(d.x-MaskDistOffset,0.0f,a.fade)/(a.fade),1.00f,4.0f);

            d.w = clampMap(mask1*df-pdf, cs.z*mask2, a.cutoff.x, a.march.x, a.march.y);
            d.w *= clampMap(d.x, 0.0, a.fade, 1.0, a.march.z);

            if(dens>0.1){
                d.w += noise2d(p_)*a.march.x*mainNoise;
            }

            pdf=df;
            p += cam_dir * d.w;
            d.x += d.w;
        }
    }
        if (fx.z <= 1.0) {
            float3 sun_samp_pos = Game2Atm(cam_pos.z*atmoRaySize+cam_dir*d.y*atmoRaySize);
            float HGS = lerp(HGStrength,HGStrengthTop,smoothstep(0.0,1.0,saturate((cam_pos.z/a.volumeBox.y)*0.5f)));
            float ScatterPhase = PhaseFunc(suncross, HGMu)*HGS;

            float3
                c_bright = (SunLight(light, sun_samp_pos, lightDirection, earth)+ScatterPhase)*sunColor*a.brightness;



            float blend = 1.0f;
            float atmo_altitude = 1.0f-saturate((cloudHeight-cam_pos.z)/cloudHeight);
            float sky_blend = lerp(max(fx.z,skyAtmoBlendClose),max(fx.z,skyAtmoBlendFar),clamp(pow( (d.y)/((skyAtmoStartDistance)*1000),skyAtmoBlendCurve*skyAtmoBlendCurve),0.0f,skyAtmoBlendMax) );
            float ground_blend = lerp(max(fx.z,groundAtmoBlendClose),max(fx.z,groundAtmoBlendFar),clamp(pow( (d.y)/((groundAtmoStartDistance)*1000),groundAtmoBlendCurve*groundAtmoBlendCurve),0.0f,groundAtmoBlendMax) );

            blend = lerp(ground_blend,sky_blend,atmo_altitude);
            blend = lerp(blend,max(blend,MyValue),pow(saturate(1.0f-p.z*cloudClipAltitudeBegin*0.001f),MyValue2)*(1.0f-atmo_altitude));

            float3 base_blend_col = lerp(b.BaseColor,bg_col,blend);
            
            float3 C;
            float3 ambi = lerp(AmbientWest,AmbientEast,saturate(suncross*ambientSplit))*saturate(exp(-fx.w)*ambientMul);
            C = c_bright * fx.x*MiePhase(suncross)*mieStrengthConfig + ((b.BaseColor+ambi)*(1.0-fx.z));
            //C = atmosphere_scattering((1.0-fx.z), C, Game2Atm(cam_pos.z), cam_dir, d.y/game2atm, lightDirection, earth);
            C = clamp(C,float3(0,0,0),clampBright.x);
            C = lerp(C,bg_col*(1.0f-fx.z),saturate(blend-fx.x*brightProtect*length(c_bright)));
            _distance = _distance*fx.z + d.y*(1.0-fx.z);

            float cross = length(lightDirection-cam_dir)*sunScale;


            //fx.z = lerp(ground_blend,sky_blend,atmo_altitude);
            //fx.z = lerp(fx.z,max(fx.z,MyValue),pow(saturate(1.0f-p.z*cloudClipAltitudeBegin*0.001f),MyValue2)*(1.0f-atmo_altitude));
            return float4(C, fx.z);
            
        }
    return float4(0.0, 0.0, 0.0 ,1.0);
}
float4 CloudsWithRays(CloudProfile a, CloudBaseColor b, const in float3 cam_dir, in float3 cam_pos, float3 light, float3 lightDirection, float Time, in out float _distance, float2 uv) {


    float3 fx = float3(0.0, 0.0, 1.0); //b, c d, t
    float suncross = dot(lightDirection, cam_dir);
	a.range.x = 1.0f/a.range.x;


    float4 dRays = float4(0.0, 0.0, 0.0, a.march.y);
        float raysBright = 0.0f;

        float3 pRays = CastStep(cam_pos, cam_dir, float2(a.volumeBox.y,a.volumeBox.y+boxOffset), dRays.x);
        dRays.y = dRays.x;
    if (dRays.x >= 0.0 && _distance>dRays.x) {
        float atm_bleed = 0.0f;      
        float last_sample = 0.0, pdf=0.0;
        float dens;
         
        for (int i = 0; fx.z > 0.0 && i < 512 && dRays.x - dRays.w < _distance && dRays.x < 100000.0f; i++) {
            float4 fmp = fakeSphereMap(pRays, cam_pos, cam_dir);
            float3 p_ = fmp.xyz;
            
            if((p_.z > a.volumeBox.y && fmp.w>10)){
                break;
            }
            else if (p_.z < a.volumeBox.y+boxOffset && fmp.w<10){
                pRays = CastStep(pRays+cam_dir*256.0f, cam_dir, float2(a.volumeBox.y,a.volumeBox.y+boxOffset), dRays.z);
                i=0;
                continue;
            }
            dens = atmoStrength; 
            float   dens_light = 
                    ShadowOnGround(
                        a ,p_, lightDirection); 

            float c_dens = (dens_light + last_sample) * (RaysminStep/2.0f);
            last_sample = dens_light;
            float df = c_dens;
                fx.x += exp((-dens_light - fx.y))*atmoStrength*(length(p_.z-(a.volumeBox.y+boxOffset))*0.00008f);   
                fx.y += fx.x*rayDensity;
                fx.z = (exp(-fx.y*(edgeSmooth*0.1f)) - 0.06f) / (1.0f - 0.06f);
                raysBright += dens_light;
                dRays.z = _distance - dRays.x;                              

            float mask1 = clamp(RMMask1-clamp(dRays.x-MaskDistOffset,0.0f,a.fade)/(a.fade),0.0f,2.0f);
            float mask2 = clamp(RMMask2+clamp(dRays.x-MaskDistOffset,0.0f,a.fade)/(a.fade),1.00f,4.0f);

            //dynamic step length
            dRays.w = clampMap(2.0f*df-pdf, 1.0f, 0.006f, 10, 10);
            dRays.w *= clampMap(dRays.x, 0.0, a.fade, 1.0, 6);

            pdf=df;

            dRays.w += noise2d(p_)*10*0.0f;

            pRays += cam_dir * dRays.w;
            dRays.x += dRays.w;
        }
    }

    float4 d = float4(0.0, 0.0, 0.0, a.march.y); 
    float3 p = CastStep(cam_pos, cam_dir, float2(a.volumeBox.x,a.volumeBox.y), d.x);
    d.y = d.x;

    if (d.x >= 0.0 && _distance>d.x) {     
        float atm_bleed = 0.0f;    
        float last_sample = 0.0, pdf=0.0;
        float suncross = dot(lightDirection, cam_dir);
        float dens;
         
        for (int i = 0; fx.z > 0.0 && i < a.march.w && d.x - d.w < _distance && d.x < a.fade; i++) {
            float4 fmp = fakeSphereMap(p, cam_pos, cam_dir);
            float3 p_ = fmp.xyz;
            
            if((p_.z > a.volumeBox.x && fmp.w>10) || p.z<cloudClipAltitudeEnd){
                break;
            }
            else if (p_.z < a.volumeBox.y && fmp.w<10){
                p = CastStep(p+cam_dir*256.0f, cam_dir, float2(a.volumeBox.x,a.volumeBox.y), d.x);
                i=0;
                continue;
            }

            float3 
                cs = CloudShape(p_.z, a.shape, a.range).xyz;

            float
                d1 = Chunk(p_, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, cs.x),
                d2 = DetailB(a,d1, p_, a.volumeBox, float3(a.densityDetail.x,a.densityDetail.y,a.densityDetail.z * (1.0f-saturate((d.x / (a.fade*0.750f) )))), a.scaleDetail, a.distortion, a.offsetC, a.offsetD,a.offsetE, cs.x),
                df = DensityField(d1, d2);
     
            if(df>cs.z+dfLimit)
            {
                
                dens = GetDensity(df, p_.z, cs.z, cs.y, a.volumeBox, a._solidness);
                float c_dens = (dens + last_sample) * (a.march.x/2.0f);
                    
                last_sample = dens;

                if (d.x >= _distance)
                    c_dens *= d.z / d.w;
                
                if(c_dens > 0.0f)
                    d.y = d.y * (1.0 - fx.z) + fx.z * d.x;
                    
                fx.y += c_dens;
                fx.z = (exp(-fx.y*(edgeSmooth*0.1f)) - a.cutoff.y) / (1.0f - a.cutoff.y);      
                d.z = _distance - d.x;
                
                float dens_light=0.0;
                if(fx.y<shadowLimit)
                {
                    dens_light = 
                    ShadowMarching(
                        c_dens ,p_, a.densityChunk, a.densityDetail,
                         a.scaleChunk, a.scaleDetail, a.cloudShift,
                          a.shape, a.range, a.offsetA, a.offsetB, a.offsetC,
                           lightDirection, float4(a.shadow.x,a.shadow.y * (1.0f-saturate((d.x / (a.fade*0.750f*6.0f) ))),a.shadow.z,a.shadow.w), a.volumeBox, a._solidness, a.distortion
                           ); 
                    
                     if(usePowder == 1){
                        fx.x += c_dens * exp((-dens_light - fx.y)) * lerp(1.0f,1.0f-(exp((-dens)*PowderExp*PowderExp)),saturate(PowderStrength*(1.0f-suncross)));
                    }else{
                        fx.x += c_dens * exp((-dens_light - fx.y));
                    }
                }
            }

            float mask1 = clamp(RMMask1-clamp(d.x-MaskDistOffset,0.0f,a.fade)/(a.fade),0.0f,2.0f);
            float mask2 = clamp(RMMask2+clamp(d.x-MaskDistOffset,0.0f,a.fade)/(a.fade),1.00f,4.0f);

            d.w = clampMap(mask1*df-pdf, cs.z*mask2, a.cutoff.x, a.march.x, a.march.y);
            d.w *= clampMap(d.x, 0.0, a.fade, 1.0, a.march.z);

            if(dens>0.1){
                d.w += noise2d(p_)*a.march.x*mainNoise;
            }

            pdf=df;
            p += cam_dir * d.w;
            d.x += d.w;
        }

    

    }
    
        

        if (fx.z <= 1.0) {
            float3 sun_samp_pos = Game2Atm(cam_pos.z+cam_dir*d.y);

            float3
                c_bright = SunLight(light, sun_samp_pos, lightDirection, earth)*sunColor*a.brightness;
                
            float ScatterPhase = PhaseFunc(suncross, HGMu);

            float HGS = lerp(HGStrength,HGStrengthTop,smoothstep(0,1,(cam_pos.z/a.volumeBox.y)*0.5f));


            float3 C;
            C = c_bright * (fx.x)*MiePhase(suncross)*mieStrengthConfig + ((ScatterPhase)*HGStrength*(fx.x))*sunColor + (b.BaseColor*(1.0-fx.x));

            C = clamp(C,float3(0,0,0),clampBright.x);
            _distance = _distance*fx.z + d.y*(1.0-fx.z);

            float cross = length(lightDirection-cam_dir)*sunScale;

            fx.z = lerp(fx.z,1.0f,clamp(pow( (d.y)/((atmoDistance2+cam_pos.z*0.025f)*1000),AtmoDistanceExp*AtmoDistanceExp),0.0f,atmoMaxBlend2) );
            return float4(C, fx.z);
            
        }
    return float4(0.0, 0.0, 0.0 ,1.0);
}
void CloudRandomness()
{
    #ifdef MOVEMENT
    cloudProfile.offsetA.xy = NOffsetA.xy*10*nsOffsetA.xy;
    cloudProfile.offsetB.xy = NOffsetA.xy*10*nsOffsetB.xy;
    cloudProfile.offsetC.xy = NOffsetA.xy*10*nsOffsetC.xy;
    cloudProfile.offsetD.xy = NOffsetA.xy*10*nsOffsetD.xy;
    //cloudProfile.offsetD.z = -abs(NOffsetA.x*3*nsOffsetD.z);
    cloudProfile.offsetE.xy = NOffsetA.xy*10*nsOffsetE.xy;
    //cloudProfile.offsetE.z = abs(NOffsetA.x*3*nsOffsetE.z);
    #endif
    
}

void CloudColorVarying(float3 SunDir, in out CloudBaseColor b)
{
    baseColor.BaseColor = _baseColor;
    baseColor.BaseColor_Day = _baseColorDay;
    baseColor.BaseColor_Sunset = _baseColorSunset;
    sunColor = ColorAtSun;     
    b.BaseColor = ColorBase;
}


float4 RenderClouds(float3 ViewDir, float3 ViewPos, float3 sunDir, float distance, float2 uv)
{

    //Bottom clouds
		/*MARCHING (ray tracing)*/ //{float4(clamp( (_cloudHeight-CameraPosition.z+_volumeShape.y*0.75f)/(_cloudHeight),1.0f,_minStep), _maxStep, _stepMult, _maxStepCount),   //min length, max length, multiplicand, max step
        /*MARCHING (ray tracing)*/ cloudProfile.march = float4( minStep, maxStep, stepMult, maxStepCount);   //min length, max length, multiplicand, max step
        /*CUTOFF*/                 cloudProfile.cutoff = float2(densCut, alphaCut);  // density cut, transparency cut
        /*VOLUME BOX*/            cloudProfile.volumeBox = float2(volumeBoxX,volumeBoxY)+cloudHeight;   //top, bottom
                                //cloudProfile.volumeBox = float2(volumeBoxX,volumeBoxY)+cloudHeight-ViewPos.z*0.25f;   //top, bottom
        /*SHAPE*/                 cloudProfile.shape = float4(float3(volumeShapeX*CLOUD_COVERAGE,volumeShapeY*CLOUD_COVERAGE,volumeShapeZ*CLOUD_COVERAGE)+cloudHeight,volumeShapeThic);       //top, mid, bot, thickness
                                    //cloudProfile.shape = float4(float3(volumeShapeX*CLOUD_COVERAGE,volumeShapeY*CLOUD_COVERAGE,volumeShapeZ*CLOUD_COVERAGE)+cloudHeight-ViewPos.z*0.25f,volumeShapeThic);       //top, mid, bot, thickness
        /*BRIGHTNESS*/            cloudProfile.brightness = cloudBrightness;
        /*RANGE*/                 //cloudProfile.range = float3(0.9 + CLOUD_COVERAGE*0.16, 0.1, 0.2+CLOUD_COVERAGE*0.4),    //total, top, bottom
        /*RANGE*/                 cloudProfile.range = float3(rangeTotal * CLOUD_COVERAGE, rangeTop, rangeBottom * CLOUD_COVERAGE),    //total, top, bottom
        /*SOLIDNESS*/             cloudProfile._solidness = float2(solidness*smoothstep(0.0,0.2,CLOUD_COVERAGE), solidnessBottom) * CLOUD_COVERAGE;   //top, bottom
        /*DENSITY CHUNK*/         cloudProfile.densityChunk = float2(densCA, densCB)*smoothstep(0.1,1.0,abs(oof));   //dens A,B
        /*SHADOW*/                cloudProfile.shadow = float4(shadowStepLength, shadowDetailStrength*0.1, shadowExpand*0.1, shadowStrength*0.1*shadowMul);  //step length, detail strength, expanding, strength
        /*DISTORTION*/            cloudProfile.distortion = float4(distMaxAngle, distStrength, distBumpStrength, distSmallBumpStrength);  //max angle, strength, bump strength, small bump strength
        /*FADE DISTANCE*/         cloudProfile.fade = fadeDist;
        /*DETAIL DENSITY*/        cloudProfile.densityDetail = float3(detailDensA,detailDensB,detailDensC);   //dens C,D,E
        /*CHUNK SCALE*/           cloudProfile.scaleChunk = float3(scaleX*0.0001, scaleY*0.001, vertS);    //scale A,B, vertical stretch
        /*DETAIL SCALE*/          cloudProfile.scaleDetail = float3(detailScaleA*0.01,detailScaleB*0.01,detailScaleC*0.01);   //scale C,D,E
        ///*CLOUD SHIFT*/            float3(-0.8, 0.0, 0.0),
        /*CLOUD SHIFT*/           cloudProfile.cloudShift = float3(-0.085, 0.0, 0.0);
        /*NOISE OFFSET A*/        cloudProfile.offsetA = nsOffsetA;
        /*NOISE OFFSET B*/        cloudProfile.offsetB =  nsOffsetB;
        /*NOISE OFFSET C*/        cloudProfile.offsetC =  nsOffsetC;
        /*NOISE OFFSET D*/        cloudProfile.offsetD =  nsOffsetD;
        /*NOISE OFFSET E*/        cloudProfile.offsetE =  nsOffsetE;
                                  cloudProfile.startDistance = 0;

                        
    CloudRandomness();
    CloudColorVarying(sunDir, baseColor);
    
    float3 lightDir = sunDir;
    float3 light = LightSource(sunDir, MoonDirection.xyz, lightDir);

    float4 cloud_temp = float4(0.0, 0.0, 0.0, 0.0);
    float4 clouds = float4(0.0, 0.0, 0.0, 0.0);
    float4 cloudGodRays = float4(0.0,0.0,0.0,0.0);

    float3 bg_col = tex2Dlod(ReShade::BackBuffer,float4(uv,0,0)).xyz;

        float depth = distance;

        clouds = CloudAtRay(bg_col,cloudProfile, baseColor, normalize(ViewDir), ViewPos, light, normalize(lightDir), timer*0.0001f, depth, uv);

    //clouds.rgb+=bg_col*clouds.w;
  
    return clouds;
}
float4 RenderCloudsWithRays(float3 ViewDir, float3 ViewPos, float3 sunDir, float distance, float2 uv)
{

    //Bottom clouds
		/*MARCHING (ray tracing)*/ //{float4(clamp( (_cloudHeight-CameraPosition.z+_volumeShape.y*0.75f)/(_cloudHeight),1.0f,_minStep), _maxStep, _stepMult, _maxStepCount),   //min length, max length, multiplicand, max step
        /*MARCHING (ray tracing)*/ cloudProfile.march = float4( minStep, maxStep, stepMult, maxStepCount);   //min length, max length, multiplicand, max step
        /*CUTOFF*/                 cloudProfile.cutoff = float2(densCut, alphaCut);  // density cut, transparency cut
        /*VOLUME BOX*/            //cloudProfile.volumeBox = float2(volumeBoxX,volumeBoxY)+cloudHeight-ViewPos.z*0.25f;   //top, bottom
                                     cloudProfile.volumeBox = float2(volumeBoxX,volumeBoxY)+cloudHeight;   //top, bottom
        /*SHAPE*/                 cloudProfile.shape = float4(float3(volumeShapeX*CLOUD_COVERAGE,volumeShapeY*CLOUD_COVERAGE,volumeShapeZ*CLOUD_COVERAGE)+cloudHeight-ViewPos.z*0.25f,volumeShapeThic);       //top, mid, bot, thickness
        /*BRIGHTNESS*/            cloudProfile.brightness = cloudBrightness;
        /*RANGE*/                 //cloudProfile.range = float3(0.9 + CLOUD_COVERAGE*0.16, 0.1, 0.2+CLOUD_COVERAGE*0.4),    //total, top, bottom
        /*RANGE*/                 cloudProfile.range = float3(rangeTotal * CLOUD_COVERAGE, rangeTop, rangeBottom * CLOUD_COVERAGE),    //total, top, bottom
        /*SOLIDNESS*/             cloudProfile._solidness = float2(solidness*smoothstep(0.0,0.2,CLOUD_COVERAGE), solidnessBottom) * CLOUD_COVERAGE;   //top, bottom
        /*DENSITY CHUNK*/         cloudProfile.densityChunk = float2(densCA, densCB);   //dens A,B
        /*SHADOW*/                cloudProfile.shadow = float4(shadowStepLength, shadowDetailStrength*0.1, shadowExpand*0.1, shadowStrength*0.1*shadowMul);  //step length, detail strength, expanding, strength
        /*DISTORTION*/            cloudProfile.distortion = float4(distMaxAngle, distStrength, distBumpStrength, distSmallBumpStrength);  //max angle, strength, bump strength, small bump strength
        /*FADE DISTANCE*/         cloudProfile.fade = fadeDist;
        /*DETAIL DENSITY*/        cloudProfile.densityDetail = float3(detailDensA,detailDensB,detailDensC);   //dens C,D,E
        /*CHUNK SCALE*/           cloudProfile.scaleChunk = float3(scaleX*0.0001, scaleY*0.001, vertS);    //scale A,B, vertical stretch
        /*DETAIL SCALE*/          cloudProfile.scaleDetail = float3(detailScaleA*0.01,detailScaleB*0.01,detailScaleC*0.01);   //scale C,D,E
        ///*CLOUD SHIFT*/            float3(-0.8, 0.0, 0.0),
        /*CLOUD SHIFT*/           cloudProfile.cloudShift = float3(-0.085, 0.0, 0.0);
        /*NOISE OFFSET A*/        cloudProfile.offsetA = nsOffsetA;
        /*NOISE OFFSET B*/        cloudProfile.offsetB =  nsOffsetB;
        /*NOISE OFFSET C*/        cloudProfile.offsetC =  nsOffsetC;
        /*NOISE OFFSET D*/        cloudProfile.offsetD =  nsOffsetD;
        /*NOISE OFFSET E*/        cloudProfile.offsetE =  nsOffsetE;
                                  cloudProfile.startDistance = 0;

                                  


    CloudRandomness();
    CloudColorVarying(sunDir, baseColor);
    

    
    float3 lightDir = sunDir;
    float3 light = LightSource(sunDir, MoonDirection.xyz, lightDir);

    

    float4 cloud_temp = float4(0.0, 0.0, 0.0, 0.0);
    float4 clouds = float4(0.0,0.0,0.0,0.0);
    float4 cloudGodRays = float4(0.0,0.0,0.0,0.0);

        float depth = distance;
        clouds = CloudsWithRays(cloudProfile, baseColor, normalize(ViewDir), ViewPos, light, normalize(lightDir), timer*0.0001f, depth, uv);
    clouds.rgb*=1.0f-clouds.w*0.65f;
  
    return clouds;
}

static const float4 gaussKernel3x3[9] =
{
  float4(-1.0, -1.0, 0.0,  1.0 / 16.0),
  float4(-1.0,  0.0, 0.0,  2.0 / 16.0),
  float4(-1.0, +1.0, 0.0,  1.0 / 16.0),
  float4( 0.0, -1.0, 0.0,  2.0 / 16.0),
  float4( 0.0,  0.0, 0.0,  4.0 / 16.0),
  float4( 0.0, +1.0, 0.0,  2.0 / 16.0),
  float4(+1.0, -1.0, 0.0,  1.0 / 16.0),
  float4(+1.0,  0.0, 0.0,  2.0 / 16.0),
  float4(+1.0, +1.0, 0.0,  1.0 / 16.0)
};

float3 ViewRay(float2 texCoord)
{
    float x = texCoord.x * 2 - 1;
    x*=-1;
    float y = (1-texCoord.y) * 2 - 1;
    float3 cameraSpaceRay = float3(x / Projection[0].x,(y / Projection[1].y)*-1, 1.0);
    float3 ray = normalize(mul(cameraSpaceRay, transpose(float3x3(WorldView[0].xyz,WorldView[1].xyz,WorldView[2].xyz))));
    return ray;
}
float4 DepthToViewPos( float depth, float2 texCoord )
{
    float  x = texCoord.x * 2 - 1;
    x*=-1;
    float  y = ( texCoord.y ) * 2 - 1;
    float2 screenSpaceRay =
        float2( (x / Projection[0].x), (y / Projection[1].y) );
    float4 pos = float4( screenSpaceRay * depth, depth, 1.0 );
    return pos;
}
float3 DepthToWorldPos( float depth, float2 texCoord )
{
    float4 pos = DepthToViewPos( depth, texCoord );
    pos        = mul( pos, InverseView );
    return pos.xyz;
}
float4 PS_Clouds(float4 vpos : SV_Position, float2 tex : TexCoord) : SV_Target
{
    float2 uv = tex;
    float2 resolution = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float3 camPos = float3(InverseView[3].x, InverseView[3].y, InverseView[3].z);


	float2 fuv = -uv;

    float4 cl = 0.0f;

    float2 texel = 1.0 / resolution;

    float3 worldPos = DepthToWorldPos(CustomDepth(uv),uv);
    float dp = length(camPos-worldPos);

    float depth = dp;



    
    float depthTest = 0.0;
    for(int i = 0;i<9;i++){
        depthTest = dp;
        if(depthTest < depth){
            depth = depthTest;
        }
    }

	float3 camDir = float3(-InverseView[0].x, -InverseView[2].z, InverseView[0].y);

    

    float3 _moonDir;
        _moonDir.x = MoonDirection.x;
        _moonDir.y = MoonDirection.y;
        _moonDir.z = MoonDirection.z;


    float3 _sunDir;
        _sunDir.x = SunDirection.x;
        _sunDir.y = SunDirection.y;
        _sunDir.z = SunDirection.z;

    float3 rayDir = normalize(worldPos - camPos)*-1;



    cl = RenderClouds(rayDir, camPos, normalize(_sunDir), depth,uv);
    
    cl.w = saturate(cl.w);
    return cl;
}
float4 PS_CloudsWithRays(float4 vpos : SV_Position, float2 tex : TexCoord) : SV_Target
{
    float2 uv = tex;
    float2 resolution = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float3 camPos = float3(InverseView[3].x, InverseView[3].y, InverseView[3].z);


	float2 fuv = -uv;

    float4 cl = 0.0f;

    float2 texel = 1.0 / resolution;

    float3 worldPos = DepthToWorldPos(CustomDepth(uv),uv);
    float dp = length(camPos-worldPos);

    float depth = dp;

    float depthTest = 0.0;
    for(int i = 0;i<9;i++){
        depthTest = dp;
        if(depthTest < depth){
            depth = depthTest;
        }
    }

	float3 camDir = float3(-InverseView[0].x, -InverseView[2].z, InverseView[0].y);


    float3 _moonDir;
        _moonDir.x = MoonDirection.x;
        _moonDir.y = MoonDirection.y;
        _moonDir.z = MoonDirection.z;

    float3 _sunDir;
        _sunDir.x = SunDirection.x;
        _sunDir.y = SunDirection.y;
        _sunDir.z = SunDirection.z;

    float3 rayDir = normalize(worldPos - camPos)*-1;

    cl = RenderCloudsWithRays(rayDir, camPos, normalize(_sunDir), depth,uv);
    cl.w = saturate(cl.w);
    return cl;
}

float4 PS_EdgeMask(float4 vpos : SV_Position, float2 tex : TexCoord) : SV_Target
{   
    float2 texel = 1.0 / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float3 camPos = float3(InverseView[3].x, InverseView[3].y, InverseView[3].z);

    float depthTest = 0.0;
    float depth = length(camPos-DepthToWorldPos(CustomDepth(tex),tex));
    float fillDepth = depth;
    for(int i = 0;i<9;i++){
        depthTest = length(camPos-DepthToWorldPos(CustomDepth(tex + texel * gaussKernel3x3[i].xy),tex)).x;
        if(depthTest > depth){
            depth = depthTest;
        }
    }

    if(length(depth-fillDepth) >= (depthFill*2.0f)*depth*3.0 && fillDepth < 125000){
        return float4(1,0,0,1);
    }
    return float4(0,0,0,0);
}
float PS_EdgeMaskGrow(float4 vpos : SV_Position, float2 tex : TexCoord) : SV_Target
{   
    float2 texel = 1.0 / float2(BUFFER_WIDTH, BUFFER_HEIGHT);

    float mask = tex2D(RTSamplerFullMask,tex.xy).x;

    for(int i=1;i<=6; i++)
    {
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(i, 0)).x);
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(0, i)).x);
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(-i, 0)).x);
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(0, -i)).x);
    } 
    return mask;
}
float4 PS_CloudsFillGap(float4 vpos : SV_Position, float2 tex : TexCoord) : SV_Target
{
    float3 camPos = float3(InverseView[3].x, InverseView[3].y, InverseView[3].z);
    if(CLOUD_COVERAGE <= 0.019 ){return float4(0.0,0.0,0.0,1.0);}

    float2 uv = tex;
    float2 resolution = float2(BUFFER_WIDTH, BUFFER_HEIGHT);

    float4 cl = 0.0f;

    float2 texel = 1.0 / resolution;


    float3 worldPos = DepthToWorldPos(CustomDepth(uv),uv);
    float dp = length(camPos-worldPos);

    float depth = dp;


    float mask = tex2D(RTSamplerFullMask,tex.xy).x;

    for(int i=1;i<=8; i++)
    {
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(i, 0)).x);
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(0, i)).x);
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(-i, 0)).x);
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(0, -i)).x);
    } 


    if(mask == 1){

	float3 camDir = float3(-InverseView[0].x, -InverseView[2].z, InverseView[0].y);

    float3 ro = camPos;
    

    

    float3 _moonDir;
        _moonDir.x = MoonDirection.x;
        _moonDir.y = MoonDirection.y;
        _moonDir.z = MoonDirection.z;


    float3 _sunDir;
        _sunDir.x = SunDirection.x;
        _sunDir.y = SunDirection.y;
        _sunDir.z = SunDirection.z;

    float3 rayDir = normalize(worldPos - camPos)*-1;

    cl = RenderClouds(rayDir, camPos, normalize(_sunDir), dp,uv);

    cl.w -= 0.00001f;
    return cl;
    }else{
        return float4(0,0,0,1);
    }
}
float4 PS_CloudsFillGapWithRays(float4 vpos : SV_Position, float2 tex : TexCoord) : SV_Target
{
    float3 camPos = float3(InverseView[3].x, InverseView[3].y, InverseView[3].z);
    if(CLOUD_COVERAGE <= 0.019 ){return float4(0.0,0.0,0.0,1.0);}

    float2 uv = tex;
    float2 resolution = float2(BUFFER_WIDTH, BUFFER_HEIGHT);

    float4 cl = 0.0f;

    float2 texel = 1.0 / resolution;


    float3 worldPos = DepthToWorldPos(CustomDepth(uv),uv);
    float dp = length(camPos-worldPos);

    float depth = dp;


    float mask = tex2D(RTSamplerFullMask,tex.xy).x;

    for(int i=1;i<=8; i++)
    {
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(i, 0)).x);
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(0, i)).x);
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(-i, 0)).x);
        mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * float2(0, -i)).x);
    } 

    if(mask == 1){

	float3 camDir = float3(-InverseView[0].x, -InverseView[2].z, InverseView[0].y);

    float3 ro = camPos;
    

    float3 _moonDir;
        _moonDir.x = MoonDirection.x;
        _moonDir.y = MoonDirection.y;
        _moonDir.z = MoonDirection.z;


    float3 _sunDir;
        _sunDir.x = SunDirection.x;
        _sunDir.y = SunDirection.y;
        _sunDir.z = SunDirection.z;

    float3 rayDir = normalize(worldPos - camPos)*-1;

    cl = RenderCloudsWithRays(rayDir, camPos, normalize(_sunDir), dp,uv);
    cl.w -= 0.00001f;
    return cl;
    }else{
        return float4(0,0,0,1);
    }
}

float4 smartDeNoise(sampler2D tex,float2 res, float2 uv, float sigma, float kSigma, float threshold)
{
    float radius = round(kSigma*sigma);
    float radQ = radius * radius;
    
    float invSigmaQx2 = .5 / (sigma * sigma);      // 1.0 / (sigma^2 * 2.0)
    float invSigmaQx2PI = INV_PI * invSigmaQx2;    // 1.0 / (sqrt(PI) * sigma)
    
    float invThresholdSqx2 = .5 / (threshold * threshold);     // 1.0 / (sigma^2 * 2.0)
    float invThresholdSqrt2PI = INV_SQRT_OF_2PI / threshold;   // 1.0 / (sqrt(2*PI) * sigma)
    
    float4 centrPx = tex2D(tex,uv);
    
    float zBuff = 0.0;
    float4 aBuff = float4(0.0,0.0,0.0,0.0);
    float2 size = res;
    
    for(float x=-radius; x <= radius && x < 64; x++) {
        float pt = sqrt(radQ-x*x);  // pt = yRadius: have circular trend
        for(float y=-pt; y <= pt && y < 64; y++) {
            float2 d = float2(x,y);

            float blurFactor = exp( -dot(d , d) * invSigmaQx2 ) * invSigmaQx2PI; 
            
            float4 walkPx =  tex2D(tex,uv+d/size);

            float4 dC = walkPx-centrPx;
            float deltaFactor = exp( -dot(dC, dC) * invThresholdSqx2) * invThresholdSqrt2PI * blurFactor;
                                 
            zBuff += deltaFactor;
            aBuff += deltaFactor*walkPx;
        }
    }
    return aBuff/zBuff;
}

float4 PS_CloudsTAA(float4 vpos : SV_Position, float2 uv : TexCoord) : SV_Target
{   
    if(CLOUD_COVERAGE <= 0.019){return float4(0.0,0.0,0.0,0.0);}
    const float2 res = float2(512,512);
    float4 _texture;
        if(deNoiseLevel == 0){
            _texture = tex2D(RTSampler512,uv);
        }else if(deNoiseLevel == 1){
            _texture = smartDeNoise(RTSampler512,res,uv,2.0,2.0,0.05);
        }else if(deNoiseLevel == 2){
            _texture = smartDeNoise(RTSampler512,res,uv,2.0,2.0,0.1);
        }else if(deNoiseLevel == 3){
            _texture = smartDeNoise(RTSampler512,res,uv,2.0,2.0,0.25);
        }

    return _texture;
}
float4 PS_CloudsTAAHD(float4 vpos : SV_Position, float2 uv : TexCoord) : SV_Target
{   
    if(CLOUD_COVERAGE <= 0.019){return float4(0.0,0.0,0.0,0.0);}
    const float2 res = float2(512,512);
    float4 _texture;
        if(deNoiseLevel == 0){
            _texture = tex2D(RTSampler1024,uv);
        }else if(deNoiseLevel == 1){
            _texture = smartDeNoise(RTSampler1024,res,uv,2.0,2.0,0.05);
        }else if(deNoiseLevel == 2){
            _texture = smartDeNoise(RTSampler1024,res,uv,2.0,2.0,0.1);
        }else if(deNoiseLevel == 3){
            _texture = smartDeNoise(RTSampler1024,res,uv,2.0,2.0,0.25);
        }

    return _texture;
}
float4 PS_CloudsTAAFull(float4 vpos : SV_Position, float2 uv : TexCoord) : SV_Target
{   
    if(CLOUD_COVERAGE <= 0.019){return float4(0.0,0.0,0.0,0.0);}
    const float2 res = float2(512,512);
    float4 _texture;
        if(deNoiseLevel == 0){
            _texture = tex2D(RTSamplerFull,uv);
        }else if(deNoiseLevel == 1){
            _texture = smartDeNoise(RTSamplerFull,res,uv,2.0,2.0,0.05);
        }else if(deNoiseLevel == 2){
            _texture = smartDeNoise(RTSamplerFull,res,uv,2.0,2.0,0.1);
        }else if(deNoiseLevel == 3){
            _texture = smartDeNoise(RTSamplerFull,res,uv,2.0,2.0,0.25);
        }

    return _texture;
}



float4 PS_CloudsSharpen(float4 vpos : SV_Position, float2 uv : TexCoord) : SV_Target
{   
    if(CLOUD_COVERAGE <= 0.019){return float4(0.0,0.0,0.0,0.0);}
	float2 resolution = float2(512, 512);
    float2 step = 1.0 / resolution.xy;	
	float3 texA = tex2Dlod(RTSampler512_2,float4(uv + float2(-step.x, -step.y) * 1.5,0.0,0.0)).rgb;
	float3 texB = tex2Dlod(RTSampler512_2,float4(uv + float2( step.x, -step.y) * 1.5,0.0,0.0)).rgb;
	float3 texC = tex2Dlod(RTSampler512_2,float4(uv + float2(-step.x,  step.y) * 1.5,0.0,0.0)).rgb;
	float3 texD = tex2Dlod(RTSampler512_2,float4(uv + float2( step.x,  step.y) * 1.5,0.0,0.0)).rgb;  
    float3 around = 0.25 * (texA + texB + texC + texD);
	float4 center  = tex2Dlod(RTSampler512_2,float4(uv,0.0,0.0));
	float3 col = center.xyz + (center.xyz - around) * cloudsSharpen;	
    return float4(col,center.w);
}
float4 PS_CloudsSharpenHD(float4 vpos : SV_Position, float2 uv : TexCoord) : SV_Target
{   
    if(CLOUD_COVERAGE <= 0.019){return float4(0.0,0.0,0.0,0.0);}
	float2 resolution = float2(1024, 1024);
    float2 step = 1.0 / resolution.xy;	
	float3 texA = tex2Dlod(RTSampler1024_2,float4(uv + float2(-step.x, -step.y) * 1.5,0.0,0.0)).rgb;
	float3 texB = tex2Dlod(RTSampler1024_2,float4(uv + float2( step.x, -step.y) * 1.5,0.0,0.0)).rgb;
	float3 texC = tex2Dlod(RTSampler1024_2,float4(uv + float2(-step.x,  step.y) * 1.5,0.0,0.0)).rgb;
	float3 texD = tex2Dlod(RTSampler1024_2,float4(uv + float2( step.x,  step.y) * 1.5,0.0,0.0)).rgb;  
    float3 around = 0.25 * (texA + texB + texC + texD);
	float4 center  = tex2Dlod(RTSampler1024_2,float4(uv,0.0,0.0));
	float3 col = center.xyz + (center.xyz - around) * cloudsSharpen;	
    return float4(col,center.w);
}
float4 PS_CloudsCA(sampler2D tex, float2 uv)
{   
    if(CLOUD_COVERAGE <= 0.019){return float4(0.0,0.0,0.0,0.0);}
	float2 resolution = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float4 ca = float4(0,0,0,0); //CA
	ca.x = tex2Dlod(tex,float4(uv-float2(0.001,0.001)*cloudsAberration,0.0,0.0)).x;
	ca.y = tex2Dlod(tex,float4(uv-float2(-0.001,-0.001)*cloudsAberration,0.0,0.0)).y;
	ca.z = tex2Dlod(tex,float4(uv-float2(0.0005,-0.0005)*cloudsAberration,0.0,0.0)).z;
	ca.w = tex2Dlod(tex,float4(uv,0.0,0.0)).w;
    return ca;
}

float4 PS_FinalDraw(float4 vpos : SV_Position, float2 uv : TexCoord) : SV_Target
{
    float depth = ReShade::GetLinearizedDepth(uv).x;
    float3 col = tex2Dlod(ReShade::BackBuffer,float4(uv,0,0)).xyz;
    float4 cl = tex2Dlod(RTSampler512_2,float4(uv,0.0,0.0));
    float4 cl_fill = tex2Dlod(RTSamplerFull,float4(uv,0.0,0.0));

    if(tex2D(RTSamplerFullMaskGrow,uv).x == 1){
        cl = cl_fill;
    }; //So many conditions I am so sorry
    
    if(length(col/3)<0.000001f+cableShit*0.0001f && depth > 0.95f && GameTime.x > 4 && GameTime.x < 21){
        cl.w = 1.0;
    }
    

    return cl;
}

float4 PS_FinalDrawHD(float4 vpos : SV_Position, float2 uv : TexCoord) : SV_Target
{
    float4 cl = tex2Dlod(RTSampler1024_2,float4(uv,0.0,0.0));
    float4 cl_fill = tex2Dlod(RTSamplerFull,float4(uv,0.0,0.0));

    if(tex2D(RTSamplerFullMaskGrow,uv).x == 1){
        cl = cl_fill;
    }; //So many conditions I am so sorry
    return cl;
}

float4 PS_Final(float4 vpos : SV_Position, float2 uv :TexCoord) : SV_Target
{
    float4 col = tex2Dlod(ReShade::BackBuffer, float4(uv,0.0,0.0));
    if(CLOUD_COVERAGE <= 0.019){return col;}

    //float4 cl = PS_CloudsCA(RTSamplerFull2,uv);
    float4 cl = tex2D(RTSamplerFull2,uv);

    if(length(cl.xyz) > 0){

		cl.xyz = cl.xyz+col.xyz*saturate(cl.w*AlphaBlend);
		col.xyz = lerp(col.xyz,cl.xyz,saturate(1.0f-cl.w));
	}
    return col;
}

technique NveClouds < ui_label = "NaturalVision Evolved: Volumetric Clouds"; hidden=true;>
{   
    pass p0
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_EdgeMask;

        RenderTarget = RenderTargetFullMask;

    }
    pass p1
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_EdgeMaskGrow;

        RenderTarget = RenderTargetFullMaskGrow;

    }
    
	pass p2
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_Clouds;

        RenderTarget = RenderTarget512;

    }
    pass p3
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_CloudsTAA;

        RenderTarget = RenderTarget512_2;

    }
    pass p4
    {
        VertexShader = PostProcessVS;
		PixelShader = PS_CloudsFillGap;

        RenderTarget = RenderTargetFull;
    } 
    
    pass p5
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_FinalDraw;

        RenderTarget = RenderTargetFull2;
    }
    pass p6
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_Final;


    }
}


technique NveCloudsHD < ui_label = "NaturalVision Evolved: Volumetric Clouds"; hidden=true;>
{   
    pass p0
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_EdgeMask;

        RenderTarget = RenderTargetFullMask;

    }
    pass p1
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_EdgeMaskGrow;

        RenderTarget = RenderTargetFullMaskGrow;

    }
    
	pass p2
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_Clouds;

        RenderTarget = RenderTarget1024;

    }
    pass p3
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_CloudsTAAHD;

        RenderTarget = RenderTarget1024_2;

    }
    pass p4
    {
        VertexShader = PostProcessVS;
		PixelShader = PS_CloudsFillGap;

        RenderTarget = RenderTargetFull;
    } 
    
    pass p5
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_FinalDrawHD;

        RenderTarget = RenderTargetFull2;
    }
    pass p6
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_Final;
    }
}

technique NveCloudsFull < ui_label = "NaturalVision Evolved: Volumetric Clouds"; hidden=true;>
{   
	pass p0
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_Clouds;

        RenderTarget = RenderTargetFull;

    }
    pass p1
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_CloudsTAAFull;

        RenderTarget = RenderTargetFull2;

    }   
    pass p2
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_Final;
    }
}

technique NveCloudsRays < ui_label = "NaturalVision Evolved: Volumetric Clouds"; hidden=true;>
{
    pass p0
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_EdgeMask;

        RenderTarget = RenderTargetFullMask;

    }
    pass p1
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_EdgeMaskGrow;

        RenderTarget = RenderTargetFullMaskGrow;

    }
    
	pass p2
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_CloudsWithRays;

        RenderTarget = RenderTarget512;

    }
    pass p3
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_CloudsTAA;

        RenderTarget = RenderTarget512_2;

    }
    pass p4
    {
        VertexShader = PostProcessVS;
		PixelShader = PS_CloudsFillGapWithRays;

        RenderTarget = RenderTargetFull;
    } 
    
    pass p5
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_FinalDraw;

        RenderTarget = RenderTargetFull2;
    }
    pass p6
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_Final;


    }

}
