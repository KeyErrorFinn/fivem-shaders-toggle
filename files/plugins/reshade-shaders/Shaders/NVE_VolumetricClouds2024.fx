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
#define LIGHTNING
//#define SECONDARY_LAYER

uniform float CloudyCoverage < hidden = true; > = 0.0f;

uniform float timer < source = "timer";>;
uniform int framecount < source = "framecount"; >;

uniform float FOG_Curve <	ui_type = "drag";ui_min = -0.0f;ui_max = 1000.0f;hidden = true; > = 1.0f;
uniform float FOG_Start <	ui_type = "drag";ui_min = -0.0f;ui_max = 1000.0f;hidden = true;> = 1.0f;
uniform float FOG_End <	ui_type = "drag";ui_min = -.0f;ui_max = 1000.0f;hidden = true;> = 1.0f;
uniform float FOG_Min <	ui_type = "drag";ui_min = -0.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float FOG_Max <	ui_type = "drag";ui_min = -.0f;ui_max = 10.0f;hidden = true;> = 1.0f;

uniform float atmCondense <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float rayleighDec <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float rayleighScatter <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float rayleighStr <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float fogDense <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float lightDec <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float mieDec <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;
uniform float adjust <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;hidden = true;> = 1.0f;

uniform float snoise <	ui_type = "drag";ui_min = -10.0f;ui_max = 10.0f;hidden = true;> = 1.0f;



/* uniform float MyValue <	ui_type = "drag";ui_min = -1000.0f;ui_max = 1000.0f;> = 1.0f;
uniform float MyValue2 <	ui_type = "drag";ui_min = -1000.0f;ui_max = 1000.0f;> = 1.0f;
uniform float MyValue3 <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;> = 1.0f;
uniform float MyValue4 <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;> = 1.0f;
uniform float MyValue5 <	ui_type = "drag";ui_min = -1.0f;ui_max = 10.0f;> = 1.0f;
uniform float3 LightningColor1 <	ui_type = "color";ui_min = 0.0f;ui_max = 10.0f;> = 1.0f;
uniform float3 LightningColor2 <	ui_type = "color";ui_min = 0.0f;ui_max = 10.0f;> = 1.0f; */

/* uniform float3 AnvilPos <	ui_type = "drag";ui_min = -10000.0f;ui_max = 10000.0f;ui_step= 5.0f;> = 1.0f;
uniform float3 AnvilShape <	ui_type = "drag";ui_min = -10000.0f;ui_max = 10000.0f;ui_step= 5.0f;> = 1.0f;
uniform float AnvilScale <	ui_type = "drag";ui_min = -100.0f;ui_max = 100.0f;> = 1.0f;
uniform float AnvilVertScale <	ui_type = "drag";ui_min = -100.0f;ui_max = 100.0f;> = 1.0f; */

/* uniform float AmbientExponent <	ui_type = "drag";ui_min = 0.0f;ui_max = 10.0f;> = 1.0f;
uniform float AmbientTop <	ui_type = "drag";ui_min = -3000.0f;ui_max = 3000.0f;ui_step = 5.0f;> = 1.0f;
uniform float AmbientBot <	ui_type = "drag";ui_min = -3000.0f;ui_max = 3000.0f;ui_step = 5.0f;> = 1.0f;
uniform float InScatterFactor <	ui_type = "drag";ui_min = 0.0f;ui_max = 10.0f;> = 1.0f;
uniform float InScatterPow <	ui_type = "drag";ui_min = 0.0f;ui_max = 10.0f;> = 1.0f;
uniform float3 InScatterColor <	ui_type = "color";ui_min = 0.0f;ui_max = 1.0f;> = 1.0f; */

/*
uniform float earthShad1 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;> = 1.0f;
uniform float earthShad2 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;> = 1.0f;
uniform float earthShad3 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;> = 1.0f;
uniform float earthShad4 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;> = 1.0f;
uniform float earthShad5 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;> = 1.0f;
uniform float earthShad6 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;> = 1.0f;
uniform float earthShad7 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;> = 1.0f;
uniform float earthShad8 <	ui_type = "drag";ui_min = -1.0f;ui_max = 1.0f;> = 1.0f;
*/

/* uniform float edgeMaskSize <	ui_type = "drag";ui_min = -5.0f;ui_max = 5.0f;> = 1.0f;
uniform float edgeMaskSizeFill <	ui_type = "drag";ui_min = -5.0f;ui_max = 5.0f;> = 1.0f;
uniform float edgeFirstSize <	ui_type = "drag";ui_min = -5.0f;ui_max = 5.0f;> = 1.0f; */


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

/*
uniform float fUIFarPlane <
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
	> = RESHADE_DEPTH_INPUT_IS_REVERSED;
*/


/* float GetLinearDepth(float2 texcoord)
{       
	float depth = tex2Dlod(ReShade::DepthBuffer, float4(texcoord, 0, 0)).x * FOG_Start;
	const float C = 0.01;

	depth = 1.0 - depth;

	const float N = 1.0;
	depth /= FOG_End*10.0 - depth * (FOG_End*10.0);
    if (depth > FOG_Curve*0.01){return 0.0;}
	return depth;	
} */

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

    float dynamicDepth = (2.5f-pow(abs(texcoord * 2.0 - 1.0),1.5)); //pseudo fix depth fov
    depth = min(1.0/depth,100000);

    return depth-(dynamicDepth*0.08)*depth;
}

float map(float x, float a, float b, float c, float d) {
    return (x - b) / (a - b) * (c - d) + d;
}
float clampMap(float x, float a, float b, float c, float d) {
    return saturate((x - b) / (a - b)) * (c - d) + d;
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

#define HASHSCALE 0.1031

float hash2(float p){
    float3 p3  = frac(float3(p,p,p) * HASHSCALE);
    p3 += dot(p3, p3.yzx + 19.19);
    return frac((p3.x + p3.y) * p3.z);
}

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

float2 rotateUV(float2 uv, float2 pivot, float rotation, float mod, float mod2)
{
    //rotation *= mod2;

    float2x2 rotation_matrix=float2x2(  float2(sin(rotation),-cos(rotation)),
                                float2(cos(rotation),sin(rotation))
                                );
    uv -= pivot;
    uv= mul(uv,rotation_matrix)*mod;
    uv += pivot;
    return uv;
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
/*
float sampledProfile(float3 p){;
    p*=0.1f;
    p.z-=AnvilPos.z;

    float pos = saturate(1.0f-length(AnvilPos.xy - p.xy)/(AnvilScale*100));
    float val = tex2Dlod(Sampler2_profile,float4(pos,clampMap(p.z*AnvilVertScale,AnvilShape.x,AnvilShape.y,0.0,1.0),0.0,0.0));
	
	return val;
	
}
*/

//=================Structs========================
struct CloudProfile
{
    float4 march;   //min length, max length, multiplicand, max step
    
    float2 cutoff;  // density cut, transparency cut

    float2 volumeBox;   //top, bottom
    float4 shape;       //top, mid, bot, thickness
    float soft;

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
float3 Distortion(float lump, float4 distortion)
{
    return float3(cos(lump*distortion.x)*distortion.y, 0.0, -lump*distortion.z);
}
float4 CloudShape(float z, float4 shape, float3 range,float softM) {
    float soft = map(z, shape.y, shape.x, range.z, range.y)*softM;

    return saturate(float4(
        smoothstep(shape.z, lerp(shape.y, shape.z, shape.w), z) * smoothstep(shape.x, lerp(shape.y, shape.x, shape.w), z), 
        range.x + soft,
        range.x - soft,
        soft));
}

float rescale(float vMin,float vMax,float v){
    return saturate((v - vMin) / (vMax - vMin));
}

float Chunk(in float3 pos, const in float2 density, const in float3 scale, const in float3 cloud_shift, const in float3 offsetA, const in float3 offsetB, in float cs) {
 
    float3 pp = pos;
    pos += cloud_shift * pos.z;

    float3
        pA = (pos + offsetA),
        pB = (pos + offsetB);

        pA.xy *= scale.x;
        pB.xy *= scale.y;
        pB.z *= chunkBZScale*0.001f;
        pA.z *= scale.z*0.001f;
  
   float dens_a = (density.x * sampledNoise3D_3DCurl(pA).x) * (sampledNoise3D_3DCurl(pB).x*density.y) * cs.x;
    //dens_a += sampledProfile(pos)*cloudMapStrength;
    //dens_a -= (1.0f-sampledNoise3D_3D(pos*cloudmapScale*0.001)*cloudMapStrength);
    //float altitudeParam = smoothstep(0.1,0.8,((pp.z)-cloudProfile.volumeBox.y)/(cloudProfile.volumeBox.x - (cloudProfile.volumeBox.y+cloudmapOffset)));
    //dens_a -= lerp((sampledNoise3D_3D(cpPos)),0.0f,1.0f-cloudMapStrength);

    return saturate(dens_a);
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

    float dens = saturate(DetailA(pos, density, scale, offsetC, d));

    d.z -= dens * distortionParm.w;

    float3 pD2 = pD;
    pD2.z *= detailZScaleB*2;

    float3 pD3 = pos + offsetE;
    pD3.z *= detailZScaleC*2;

    float3 cam_pos = float3(InverseView[3].x, InverseView[3].y, InverseView[3].z);
    float dist = length(pos - cam_pos);

    float c_factor = clampMap(dist*2.0f,0.0,cp.fade,1.0,0.0);

    float b = saturate((sampledNoise3D_3D((pD2 + d/3.0) * scale.y)));
    float c = saturate(sampledNoise3D_3DCurl((pD3 + d*8.0) * scale.z));
    float cb = saturate(sampledNoise3D_3DCurl((pD3*0.5 + d*8.0) * scale.z));
    float co = saturate(density.z * saturate(pow(abs(cb*2-1),rcp(max(C_Contrast,0.0001))) * sign(cb - 0.5) + 0.5));

    float alt_factor = clampMap(pos.z,BottomDetailLow,BottomDetailHigh,BottomDetailMul,0.0);



    //dens = map(dens,0.0,1.0,0.0,1.0-saturate(c*c_factor+co*alt_factor*c_factor+b));
    //dens = rescale(saturate(b*density.y+c*density.z+c*alt_factor),MyValue3,dens);
    dens+=c*density.z+b*density.y;
    dens-=co*alt_factor*c_factor;
    return dens;
}
float DensityField(float lump, float detail)
{
    return (lump*detail+lump);
}
float GetDensity(float density_field, float height, float low, float high, float2 volumeBox, float2 _solidness)
{
    return clampMap(density_field, low, high, 0.0, clampMap(height, volumeBox.y, volumeBox.x, _solidness.y, _solidness.x));
}

//==============SCATTERING FUNCTIONS=========================
float3 pow3(float3 v, float n)
{
    return float3(pow(v.x, n), pow(v.y, n), pow(v.z, n));
}

#define atmosphereStep (15.0)
#define lightStep (3.0)
#define fix (0.00001)
#define sunLightStrength (sunStrength)
#define moonReflecticity (0.125)
#define LightingDecay (lightDec)
#define rayleighStrength (rayleighStr)
#define rayleighDecay (rayleighDec)
#define atmDensity (atmDensityConfig)
#define atmCondensitivity (atmCondense)
#define waveLengthFactor (float3(1785.0625, 850.3056, 410.0625))
#define scatteringFactor (float3(1.0, 1.0, 1.0)*waveLengthFactor/rayleighDecay)  // for tuning, should be 1.0 physically, but you know.
#define mieStrength (mieStrengthConfig + adjust*0.1)
#define mieDecay (mieDec)
#define fogDensity (fogDense + adjust)
#define earthRadius (6.57)
#define  groundHeight (6.48)
#define AtmOrigin (float3(0.0, 0.0, groundHeight))
#define earth (float4(0.0, 0.0, 0.0, earthRadius))
#define game2atm (atmoConvDist*100000.0f)
static float3 sunColor;

float3 Game2Atm(float3 gamepos)
{
    return (gamepos/(game2atm)) + AtmOrigin;
}
float3 Game2Atm_2(float3 gamepos)
{
    return float3(0.0,0.0,gamepos.z/game2atm) + AtmOrigin;
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

    float fog = fogDensity*(rcp(1200.0*h+0.5))/2.0;
    
    if(h<0.0){
        return float2(strength, fogDensity);
    }

    return float2((exp(-h*condense)-ep)/(1.0-ep)*strength, fog);
}

float3 rayleighScattering(float c)
{
    return (1.0+c*c)*rayleighStrength/waveLengthFactor;
}

float HenyeyGreenstein(float inCos, float inG){
    float num = 1.0 - inG * inG;
    float denom = 1.0 + inG * inG - 2.0 * inG * inCos;
    float rsqrt_denom = rsqrt(denom);

    return num * rsqrt_denom * rsqrt_denom * rsqrt_denom * (1.0 / (4.0 * Pi));
}

float MiePhase(float c)
{
    //return 1.0f+1.5f*exp(25.0f*(c-1.0f));
    return 1.0f+HenyeyGreenstein(c, 0.4);
}

float MieScattering(float c)
{
    return mieStrength*MiePhase(c);
}

float3 LightDecay(float densityR, float densityM)
{
    return exp(-densityR/scatteringFactor - densityM*mieDecay);
}

float3 SunLight(float3 light, float3 position, float3 lightDirection, float4 sphere)
{
    float3 smp;
    float4 sms = sphereCast(position, normalize(lightDirection), sphere, lightStep, smp);      
    float2 dl;
    
    for(float j=0.0; j<lightStep; j++)
    {
        smp+=sms.xyz;
        dl += Density(smp, sphere, atmDensity, atmCondensitivity);
    }

    return saturate(light*LightDecay(dl.x, dl.y)/LightingDecay);

}

float3 LightSource(float3 SunDir,float3 MoonDir, out float3 SourceDir)
{
    float val1 = earthShadVal1;
    float val2 = earthShadVal2;
    float val3 = earthShadVal3;
    float val4 = earthShadVal4;
    float val5 = earthShadVal5;
    float val6 = earthShadVal6;
    float val7 = earthShadVal7;

    /*
    float val1 = earthShad1;
    float val2 = earthShad2;
    float val3 = earthShad3;
    float val4 = earthShad4;
    float val5 = earthShad5;
    float val6 = earthShad6;
    float val7 = earthShad7;
    float val8 = earthShad8;
    */
    
    //const float val1 = -0.035;
    //const float val2 = 0.0;
    //const float val3 = -0.035;
    //const float val4 = 0.0;
    //const float val5 = -0.35;
    //const float val6 = -0.035;
    //const float val7 = -0.1;
    //const float val8 = 0.0;
    /* if(SunDir.x < 0.5){ //SUNSET

        if(SunDir.z<val1)
        {   
            SourceDir = MoonDir;
            return sunLightStrength*moonReflecticity*smoothstep(val2, val3, SunDir.z)*sunColor;
        }
        SourceDir = SunDir;
        return sunLightStrength*smoothstep(val3, val4, SunDir.z)*sunColor;
    }else{ 
        if(SunDir.z<val5)
            {   
            SourceDir = MoonDir;
            return sunLightStrength*moonReflecticity*smoothstep(val6, val7, SunDir.z)*sunColor;
        }
        SourceDir = SunDir;
        return sunLightStrength*smoothstep(val7, val8, SunDir.z)*sunColor;
    } */
    if(SunDir.x < 0.5){ //SUNSET

        if(SunDir.z<val1)
        {   
            SourceDir = MoonDir;
            return sunLightStrength*moonReflecticity*sunColor;
        }
        SourceDir = SunDir;
        return sunLightStrength*sunColor;
    }else{ 
        if(SunDir.z<val5)
            {   
            SourceDir = MoonDir;
            return sunLightStrength*moonReflecticity*sunColor;
        }
        SourceDir = SunDir;
        return sunLightStrength*sunColor;
    }
}


float ShadowMarching(CloudProfile cp,float dens, float3 p, float2 densityChunk, float3 densityDetail, float3 scaleChunk, float3 scaleDetail, float3 shift, float4 profile, float3 dens_thres, const in float3 offsetA, const in float3 offsetB, const in float3 offsetC, float3 SunDir, float4 shadow, float2 volumeBox, float2 _solidness, float4 distortionParm) {
    if(dens<=shadowEarlyExit*0.0001f) return dens*shadow.x*shadowEarlyExitApprox;  //return a approx. value
    
    const float threshold = shadowThreshold/shadow.w/shadow.x;
    
    float d = 0.0;
    float4 step = float4(SunDir.xyz*shadow.x, shadow.x);
    int s_t = 0;
    

    for(int i=0; p.z<volumeBox.x && p.z>volumeBox.y && i<shadowSteps; i++)
    {
        float4 cs = CloudShape(p.z, profile, dens_thres,cp.soft);
        float d1 = Chunk(p, densityChunk, scaleChunk, shift, offsetA, offsetB, cs.x);
        
        float3 displace = Distortion(d1, distortionParm);
        float d2 = DetailA(p,densityDetail, scaleDetail, offsetC, displace) * shadow.y;
        d += GetDensity(DensityField(d1, d2), p.z, cs.z-shadow.z*50.0+shadowOffset, cs.y, volumeBox, float2(0.99,0.88));
        s_t+=1;
        p+=step.xyz;

    }

    return d*shadow.w*s_t;
}
float ShadowOnGround(CloudProfile a, float3 position, float3 SunDirection)
{    
    a.range.x = rcp(a.range.x);
        
	float d;
    //float3 samppos = PosOnPlane(position, SunDirection, a.shape.y-C_fadeDist*100, d);
	float3 samppos = PosOnPlane(position, SunDirection, a.shape.y, d);

    float d1 = 0.0, d2 = 0.0;
    float4 cs;
    
	cs = CloudShape(samppos.z, a.shape, float3(a.range.x*1.0,a.range.y,a.range.z),a.soft);
	d1 = Chunk(samppos, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, cs.x);
	
	float3 displace = Distortion(d1, a.distortion);
	d2 = DetailA(samppos, a.densityDetail, a.scaleDetail, a.offsetC, displace) * a.shadow.y;    
	float adjustTerm =  max(a.shape.y - max(a.volumeBox.y, position.z*1.0f), 0.0) * clampMap(d, 0.0, a.fade, 1.0, 0.0);
    
	return GetDensity(DensityField(d1, d2), samppos.z, cs.z-a.shadow.z*2.0f, cs.y*10.0f, a.volumeBox, a._solidness)*adjustTerm;
}
float ShadowStatic(CloudProfile a, float3 position, float3 SunDirection, float mul)
{    
    a.range.x = rcp(a.range.x);
    position.z = min(position.z,ray_shape_offset);
        
	float d;
    //float3 samppos = PosOnPlane(position, SunDirection, a.shape.y-C_fadeDist*100, d);
	float3 samppos = PosOnPlane(position, SunDirection, a.shape.y, d);



    float d1 = 0.0, d2 = 0.0;
    float4 cs;
    

    float length_adjust = clampMap(d*0.1,0,a.volumeBox.x,1.0,0.0);
	cs = CloudShape(samppos.z, a.shape, float3(a.range.x*ray_shape,a.range.y,a.range.z),a.soft);
	d1 = Chunk(samppos, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, cs.x);
	
	float3 displace = Distortion(d1, a.distortion);
	d2 = DetailA(samppos, a.densityDetail, a.scaleDetail, a.offsetC, displace) * a.shadow.y;    
	float adjustTerm =  max(a.shape.y - max(a.volumeBox.y, position.z*1.0f), 0.0) * clampMap(d, 0.0, a.fade, 1.0, 0.0) * length_adjust;
    
	return GetDensity(DensityField(d1, d2), samppos.z, cs.z-a.shadow.z*2.0f, cs.y*10.0f, a.volumeBox, a._solidness)*adjustTerm*mul;
}
float CoverageIntersect(CloudProfile a, float3 position, float mul)
{    
    a.range.x = rcp(a.range.x);
        
	float d;
    //float3 samppos = PosOnPlane(position, SunDirection, a.shape.y-C_fadeDist*100, d);
	float3 samppos = PosOnPlane(position, float3(0,0.0,1.0), a.shape.y, d);


    float d1 = 0.0, d2 = 0.0;
    float4 cs;
    

    float length_adjust = clampMap(d*0.1,0,a.volumeBox.x,1.0,0.0);
	cs = CloudShape(samppos.z, a.shape, float3(a.range.x*1.0,a.range.y,a.range.z),a.soft);
	d1 = Chunk(samppos, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, cs.x);
	
	float3 displace = Distortion(d1, a.distortion);
	d2 = DetailA(samppos, a.densityDetail, a.scaleDetail, a.offsetC, displace) * a.shadow.y;    
	float adjustTerm =  max(a.shape.y - max(a.volumeBox.y, position.z*1.0f), 0.0) * clampMap(d, 0.0, a.fade, 1.0, 0.0) * length_adjust;
    
	return GetDensity(DensityField(d1, d2), samppos.z, cs.z-a.shadow.z*2.0f, cs.y*10.0f, a.volumeBox, a._solidness)*adjustTerm;
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
    marchStep.w = 15.0f*distance/atmosphereStep/game2atm;
    marchStep.xyz = ray*marchStep.w;
    
    float3 lightDirection = 0.0;
    float3 lightStrength = LightSource(SunDirection,MoonDirection.xyz, lightDirection);
    
    float3 scattered = AtmosphereScattering(color,  Game2Atm_2(camera), marchStep, ray, lightStrength, lightDirection, 1.0, sphere);
    return lerp(color, scattered, fade);
}

float4 CloudAtRay(in float3 bg_col,CloudProfile a, CloudBaseColor b, const in float3 cam_dir, in float3 cam_pos, float3 light, float3 lightDirection, float Time, in out float _distance, float2 uv) {

    float4 d = float4(0.0, 0.0, 0.0, a.march.y); 
    float3 p = CastStep(cam_pos, cam_dir, float2(a.volumeBox.x,a.volumeBox.y), d.x);

    d.y = d.x;

    float3 p_;

    if (d.x >= 0.0 && _distance>d.x) {

        a.range.x = rcp(a.range.x);

        float4 fx = float4(0.0, 0.0, 1.0,0.0); 
        float fxl = 0.0;
        float is = 0.0f;
        float h = 0.0;
        float suncross = dot(lightDirection, cam_dir);
        
        float atm_bleed = 0.0f;
        
        float last_sample = 0.0, pdf=0.0;


        float lightning = 0.0;

        float dens;

        float atmo_altitude = 1.0f-saturate((cloudHeight-cam_pos.z)/cloudHeight);
         
        for (int i = 0; fx.z > 0.015 && i < a.march.w && d.x - d.w < _distance && d.x < a.fade; i++) {
            float4 fmp = fakeSphereMap(p, cam_pos, cam_dir);
            float3 p_ = fmp.xyz;
            
            if((p_.z > a.volumeBox.x && fmp.w>10) || p.z<cloudClipAltitudeEnd){
                break;
            }
            else if (p_.z < a.volumeBox.y && fmp.w<10){
                p = CastStep(p+cam_dir, cam_dir, float2(a.volumeBox.x,a.volumeBox.y), d.x);
                i=0;
                continue;
            }

            float3 
                cs = CloudShape(p_.z, a.shape, a.range, a.soft).xyz;

            float
                d1 = Chunk(p_, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, cs.x),
                d2 = DetailB(a,d1, p_, a.volumeBox, float3(a.densityDetail.x,a.densityDetail.y,a.densityDetail.z), a.scaleDetail, a.distortion, a.offsetC, a.offsetD,a.offsetE, cs.x),
                df = DensityField(d1, d2);

            float shadow_term = 0.0;
          
            if(df>cs.z+dfLimit)
            {   
                dens = GetDensity(df, p_.z, cs.z, cs.y, a.volumeBox, a._solidness);
                float c_dens = saturate((dens + last_sample) * (a.march.x*0.1f));
                    
                last_sample = dens;

                if (d.x >= _distance)
                    c_dens *= d.z / d.w;
                
                if(c_dens > 0.0f)
                    d.y = d.y * (1.0 - fx.z) + fx.z * d.x;
                    
                fx.y += c_dens; 
                fxl += c_dens;
                fx.z *= ((1.0-c_dens*1.5)- a.cutoff.y) / (1.0f - a.cutoff.y);       
                d.z = _distance - d.x;

                float dens_light=0.0;
                if(fx.y<shadowLimit)
                {
                    dens_light = //ShadowOnGround(cloudProfile2,p_,lightDirection) +
                    ShadowMarching(a,
                        fx.y ,p_, a.densityChunk, a.densityDetail,
                         a.scaleChunk, a.scaleDetail, a.cloudShift,
                          a.shape, a.range, a.offsetA, a.offsetB, a.offsetC,
                           lightDirection, a.shadow, a.volumeBox, a._solidness, a.distortion
                           ); 
                    

                    /* fx.x +=
                    max(exp(-dens_light),exp(-dens_light*InScatterExp)*InScatterMul) * 
                    lerp(1.0f,1.0f-(exp((-dens)*PowderExp*PowderExp)),saturate(PowderStrength*(1.0f-suncross))); */

                    float powder = lerp(1.0f,1.0f-(exp((-dens)*PowderExp*PowderExp)),saturate(PowderStrength*(1.0f-suncross)));
                    float ambi_exp = lerp(1.0f,1.0f-(exp((-dens)*ambient_power*ambient_power)),saturate(ambient_factor));

                    fx.x += max(exp(-dens_light),exp(-dens_light*InScatterExp)*InScatterMul) * c_dens * fx.z * powder;

                    fx.w += exp(-c_dens-fx.y)*clampMap(p_.z,ambient_altitude_bottom,ambient_altitude_top,0,1.0)*ambi_exp;
                }

                h+= exp(-p_.z);

                #ifdef LIGHTNING

                float l_toggle = 1.0;

                float l_toggle2 = hash2(floor(Time*80.0));

                if(hash2(floor(Time*L_Frequency)) > L_Probability){
            	    l_toggle = 0.0;
                }

                float3 L_Center = float3(L_CenterX, L_CenterY, L_CenterZ);
                //Lightning
                float3 l_pos = float3(L_Center.x+(hash2(floor(Time*0.2))*2.0-1.0)*5000*l_toggle2,L_Center.y+(hash2(floor(Time*0.1))*2.0-1.0)*5000*l_toggle2,L_Center.z+hash2(floor(Time*1.5))*1500);
                //float3 l_pos = L_Center;
                
                float r_size = max(0.3,hash2(Time*L_StrobeSpeed)*rcp(L_StrobeMul)*hash2(Time*L_StrobeSpeed*0.5)*hash2(Time*L_StrobeSpeed*0.1));
                float l_size = pow(1.0f+length(p_-l_pos)*L_Size*0.0001*r_size,-L_Curve);
                lightning+=max(exp(-fxl),exp(-fxl*0.2)*0.75)*l_size*lerp(1.0f,1.0f-(exp((-dens)*PowderExp*PowderExp)),saturate(PowderStrength));
                lightning*=l_toggle;
                #endif
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
    
        if (fx.z <= 1.0) {
            //fx = saturate(fx);

            float3 z_pos = float3(0.0,0.0,cam_pos.z);
            float3 sun_samp_pos = Game2Atm(z_pos + cam_dir*d.y);
            float HGS = lerp(HGStrength,HGStrengthTop,smoothstep(0.0,1.0,saturate((cam_pos.z/a.volumeBox.y)*0.5f)));
            float ScatterPhase = HenyeyGreenstein(suncross, HGMu) * HGStrength;
            float cross = -1.0f+pow(1.0f+length(lightDirection-cam_dir)*1.5f,4.0);
            
            //float medium_height = p_.z*AmbientTop;

            float3
                c_bright = (light)*a.brightness;

            
            //c_bright *= LightDecay(fx.y,fx.x);

            //fx.x = 1.0 - exp(-fx.x);
            c_bright = 1.0 - exp(-c_bright);
            c_bright += ScatterPhase*sunColor;


            float blend = 1.0f;
            float sky_blend = lerp(max(fx.z,skyAtmoBlendClose),max(fx.z,skyAtmoBlendFar),clamp(pow( (d.y)/((skyAtmoStartDistance)*1000),skyAtmoBlendCurve*skyAtmoBlendCurve),0.0f,skyAtmoBlendMax) );
            float ground_blend = lerp(max(fx.z,groundAtmoBlendClose),max(fx.z,groundAtmoBlendFar),clamp(pow( (d.y)/((groundAtmoStartDistance)*1000),groundAtmoBlendCurve*groundAtmoBlendCurve),0.0f,groundAtmoBlendMax) );

            blend = lerp(ground_blend,sky_blend,atmo_altitude);
            // surface haze
            //blend = lerp(blend,max(blend,1.1),pow(saturate(1.0f-p.z*cloudClipAltitudeBegin*0.001f),1.1)*(1.0f-atmo_altitude));

            // limit blending around sun
            //blend = lerp(blend,0.0f,1.0f-saturate(cross));

            float3 base_blend_col = lerp(b.BaseColor,bg_col,blend);
            
            float3 C;
            C = c_bright * fx.x;
            C+=AmbientSkyColor*fx.w;
            //C += ( ambi * saturate(1.0-fx.w*AmbientExponent*fx.y));
            //C += bg_col*saturate(1.0-fx.z*AmbientExponent-exp(-fx.y));
            //C += is*InScatterColor;


            #ifdef LIGHTNING
            // Lightning
            float3 l_color = lerp(float3(0.197,0.326,0.49),float3(0.237,0.307,0.421),hash2(floor(Time)));
            C += ( l_color * lightning );
            fx.z = saturate(fx.z-lightning);
            #endif

            C += b.BaseColor*saturate(1.0f-fx.x-fx.z);
            //C = atmosphere_scattering((1.0-fx.z), C, Game2Atm(cam_pos.z), cam_dir, d.y/(game2atm*MyValue3*0.1), lightDirection, earth);
            //C = clamp(C,float3(0,0,0),clampBright.x);
            C = lerp(C,bg_col*saturate(1.0f-fx.z),saturate(blend-fx.x*brightProtect*pow(length(c_bright)*fx.z,2.0)));
            _distance = _distance*fx.z + d.y*(1.0-fx.z);




            //fx.z = lerp(ground_blend,sky_blend,atmo_altitude);
            //fx.z = lerp(fx.z,max(fx.z,MyValue),pow(saturate(1.0f-p.z*cloudClipAltitudeBegin*0.001f),MyValue2)*(1.0f-atmo_altitude));
            return float4(C, fx.z);
        }
            
    }
    return float4(0.0, 0.0, 0.0 ,1.0);
}
float4 CloudsWithRays(in float3 bg_col,CloudProfile a, CloudBaseColor b, const in float3 cam_dir, in float3 cam_pos, float3 light, float3 lightDirection, float Time, in out float _distance, float2 uv) {


    float ray_c = 0.0;
    float suncross = dot(lightDirection, cam_dir);


    float4 d = float4(0.0, 0.0, 0.0, a.march.y); 
    float3 p = CastStep(cam_pos, cam_dir, float2(a.volumeBox.x,a.volumeBox.y+boxOffset), d.x);
    d.y = d.x;

    if (d.x >= 0.0 && _distance>d.x) {    
         a.range.x = rcp(a.range.x);

        float4 fx = float4(0.0, 0.0, 1.0,0.0); 
        float fxl = 0.0;
        float is = 0.0f;
        float h = 0.0; 
        float atm_bleed = 0.0f;    
        float last_sample = 0.0, pdf=0.0;
        float dens;

        float lightning = 0.0;

        float atmo_altitude = 1.0f-saturate((cloudHeight-cam_pos.z)/cloudHeight);
         
        for (int i = 0; fx.z > a.cutoff.y && i < a.march.w && d.x - d.w < _distance && d.x < a.fade; i++) {
            float4 fmp = fakeSphereMap(p, cam_pos, cam_dir);
            float3 p_ = fmp.xyz;
            
            if((p_.z > a.volumeBox.x && fmp.w>10) || p.z<cloudClipAltitudeEnd){
                break;
            }
            else if (p_.z < a.volumeBox.y+boxOffset && fmp.w<10){
                p = CastStep(p+cam_dir, cam_dir, float2(a.volumeBox.x,a.volumeBox.y+boxOffset), d.x);
                i=0;
                continue;
            }

            float v_fade = smoothstep(ray_altitude_begin,ray_altitude_end,p_.z)*smoothstep(0,500,p.z);
            float ray_dens = pow(exp(-ShadowStatic(a ,p_, lightDirection,1.0))*rayDensity*0.001*v_fade,2.0);
            //float d3 = ShadowOnGround(a ,p_, lightDirection);

            float3 
                cs = CloudShape(p_.z, a.shape, a.range, a.soft).xyz;

            float
                d1 = Chunk(p_, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, cs.x),
                d2 = DetailB(a,d1, p_, a.volumeBox, float3(a.densityDetail.x,a.densityDetail.y,a.densityDetail.z * (1.0f-saturate((d.x / (a.fade*0.750f) )))), a.scaleDetail, a.distortion, a.offsetC, a.offsetD,a.offsetE, cs.x),
                df = DensityField(d1, d2);

     
            if(df>-1)
            {
                
                dens = GetDensity(df, p_.z, cs.z, cs.y, float2(a.volumeBox.x,a.volumeBox.y+boxOffset), a._solidness)+atmoStrength+ray_dens;
                float c_dens = (dens + last_sample) * (a.march.x*0.1f);

                    
                last_sample = dens;

                if (d.x >= _distance)
                    c_dens *= d.z / d.w;
                
                if(c_dens > 0.0f)
                    d.y = d.y * (1.0 - fx.z) + fx.z * d.x;
                    
                fx.y += c_dens; 
                fxl += c_dens;
                fx.z *= ((1.0-(c_dens)*1.5)- 0.07) / (1.0f - 0.07);     
                d.z = _distance - d.x;
                    
                
                float dens_light=0.0;
                if(fx.y<shadowLimit)
                {
                    dens_light = //ShadowOnGround(cloudProfile2,p_,lightDirection) +
                    ShadowMarching(a,
                        fx.y ,p_, a.densityChunk, a.densityDetail,
                         a.scaleChunk, a.scaleDetail, a.cloudShift,
                          a.shape, a.range, a.offsetA, a.offsetB, a.offsetC,
                           lightDirection, a.shadow, float2(a.volumeBox.x,a.volumeBox.y+boxOffset), a._solidness, a.distortion
                           ); 
                    

                    /* fx.x +=
                    max(exp(-dens_light),exp(-dens_light*InScatterExp)*InScatterMul) * 
                    lerp(1.0f,1.0f-(exp((-dens)*PowderExp*PowderExp)),saturate(PowderStrength*(1.0f-suncross))); */

                    float powder = lerp(1.0f,1.0f-(exp((-dens)*PowderExp*PowderExp)),saturate(PowderStrength*(1.0f-suncross)));
                    float ambi_exp = lerp(1.0f,1.0f-(exp((-dens)*ambient_power*ambient_power)),saturate(ambient_factor));

                    fx.x += max(exp(-dens_light),exp(-dens_light*InScatterExp)*InScatterMul) * c_dens * fx.z * powder;

                    fx.w += exp(-c_dens-fx.y)*clampMap(p_.z,ambient_altitude_bottom,ambient_altitude_top,0,1.0)*ambi_exp;
                }

                h+= exp(-p_.z);

                #ifdef LIGHTNING

                float l_toggle = 1.0;

                float l_toggle2 = hash2(floor(Time*80.0));

                if(hash2(floor(Time*L_Frequency)) > L_Probability){
            	    l_toggle = 0.0;
                }

                float3 L_Center = float3(L_CenterX, L_CenterY, L_CenterZ);
                //Lightning
                float3 l_pos = float3(L_Center.x+(hash2(floor(Time*0.2))*2.0-1.0)*5000*l_toggle2,L_Center.y+(hash2(floor(Time*0.1))*2.0-1.0)*5000*l_toggle2,L_Center.z+hash2(floor(Time*1.5))*1500);
                //float3 l_pos = L_Center;
                
                float r_size = max(0.3,hash2(Time*L_StrobeSpeed)*rcp(L_StrobeMul)*hash2(Time*L_StrobeSpeed*0.5)*hash2(Time*L_StrobeSpeed*0.1));
                float l_size = pow(1.0f+length(p_-l_pos)*L_Size*0.0001*r_size,-L_Curve);
                lightning+=max(exp(-fxl),exp(-fxl*0.2)*0.75)*l_size*lerp(1.0f,1.0f-(exp((-dens)*PowderExp*PowderExp)),saturate(PowderStrength));
                lightning*=l_toggle;
                #endif
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

    

    
          

       if (fx.z <= 1.0) {
            //fx = saturate(fx);

            float3 z_pos = float3(0.0,0.0,cam_pos.z);
            float3 sun_samp_pos = Game2Atm(z_pos + cam_dir*d.y);
            float HGS = lerp(HGStrength,HGStrengthTop,smoothstep(0.0,1.0,saturate((cam_pos.z/a.volumeBox.y)*0.5f)));
            float ScatterPhase = HenyeyGreenstein(suncross, HGMu) * HGStrength;
            float cross = -1.0f+pow(1.0f+length(lightDirection-cam_dir)*1.5f,4.0);
            
            //float medium_height = p_.z*AmbientTop;

            float3
                c_bright = (light)*a.brightness;

            
            //c_bright *= LightDecay(fx.y,fx.x);

            //fx.x = 1.0 - exp(-fx.x);
            c_bright = 1.0 - exp(-c_bright);
            c_bright += ScatterPhase*sunColor;


            float blend = 1.0f;
            float sky_blend = lerp(max(fx.z,skyAtmoBlendClose),max(fx.z,skyAtmoBlendFar),clamp(pow( (d.y)/((skyAtmoStartDistance)*1000),skyAtmoBlendCurve*skyAtmoBlendCurve),0.0f,skyAtmoBlendMax) );
            float ground_blend = lerp(max(fx.z,groundAtmoBlendClose),max(fx.z,groundAtmoBlendFar),clamp(pow( (d.y)/((groundAtmoStartDistance)*1000),groundAtmoBlendCurve*groundAtmoBlendCurve),0.0f,groundAtmoBlendMax) );

            blend = lerp(ground_blend,sky_blend,atmo_altitude);
            // surface haze
            //blend = lerp(blend,max(blend,1.1),pow(saturate(1.0f-p.z*cloudClipAltitudeBegin*0.001f),1.1)*(1.0f-atmo_altitude));

            // limit blending around sun
            //blend = lerp(blend,0.0f,1.0f-saturate(cross));

            float3 base_blend_col = lerp(b.BaseColor,bg_col,blend);
            
            float3 C;
            C = c_bright * fx.x;
            C+=AmbientSkyColor*fx.w;
            //C += ( ambi * saturate(1.0-fx.w*AmbientExponent*fx.y));
            //C += bg_col*saturate(1.0-fx.z*AmbientExponent-exp(-fx.y));
            //C += is*InScatterColor;


            #ifdef LIGHTNING
            // Lightning
            float3 l_color = lerp(float3(0.197,0.326,0.49),float3(0.237,0.307,0.421),hash2(floor(Time)));
            C += ( l_color * lightning );
            fx.z = saturate(fx.z-lightning);
            #endif

            C += b.BaseColor*saturate(1.0f-fx.x-fx.z);
            //C = atmosphere_scattering((1.0-fx.z), C, Game2Atm(cam_pos.z), cam_dir, d.y/(game2atm*MyValue3*0.1), lightDirection, earth);
            //C = clamp(C,float3(0,0,0),clampBright.x);
            C = lerp(C,bg_col*saturate(1.0f-fx.z),saturate(blend-fx.x*brightProtect*pow(length(c_bright)*fx.z,2.0)));
            _distance = _distance*fx.z + d.y*(1.0-fx.z);




            //fx.z = lerp(ground_blend,sky_blend,atmo_altitude);
            //fx.z = lerp(fx.z,max(fx.z,MyValue),pow(saturate(1.0f-p.z*cloudClipAltitudeBegin*0.001f),MyValue2)*(1.0f-atmo_altitude));
            return float4(C, fx.z);
        }
    }
    return float4(0.0, 0.0, 0.0 ,1.0);
}
void CloudRandomness()
{
    #ifdef MOVEMENT
    cloudProfile.offsetA.xy = NOffsetA.xy*5*nsOffsetA.xy;
    cloudProfile.offsetA.z = NOffsetA.z*20*nsOffsetD.z;
    cloudProfile.offsetB.xy = NOffsetA.xy*1*nsOffsetB.xy;
    cloudProfile.offsetC.xy = NOffsetA.xy*1*nsOffsetC.xy;
    cloudProfile.offsetD.xy = NOffsetA.xy*1*nsOffsetD.xy;
    //cloudProfile.offsetD.z = -abs(NOffsetA.x*3*nsOffsetD.z);
    cloudProfile.offsetE.xy = NOffsetA.xy*2*nsOffsetE.xy;
    //cloudProfile.offsetE.z = abs(NOffsetA.x*3*nsOffsetE.z);

    cloudProfile2.offsetA.xy = NOffsetA.xy*5*nsOffsetA.xy;
    cloudProfile2.offsetB.xy = NOffsetA.xy*1*nsOffsetB.xy;
    cloudProfile2.offsetC.xy = NOffsetA.xy*1*nsOffsetC.xy;
    cloudProfile2.offsetD.xy = NOffsetA.xy*1*nsOffsetD.xy*10;
    //cloudProfile.offsetD.z = -abs(NOffsetA.x*3*nsOffsetD.z);
    cloudProfile2.offsetE.xy = NOffsetA.xy*1*nsOffsetE.xy*5;
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


void updateProfiles(){
    //Bottom clouds
		/*MARCHING (ray tracing)*/ //{float4(clamp( (_cloudHeight-CameraPosition.z+_volumeShape.y*0.75f)/(_cloudHeight),1.0f,_minStep), _maxStep, _stepMult, _maxStepCount),   //min length, max length, multiplicand, max step
        /*MARCHING (ray tracing)*/ cloudProfile.march = float4( 5, 80, 25, 256);   //min length, max length, multiplicand, max step
        /*CUTOFF*/                 cloudProfile.cutoff = float2(densCut, alphaCut);  // density cut, transparency cut
        /*VOLUME BOX*/            cloudProfile.volumeBox = float2(volumeBoxX,volumeBoxY)+cloudHeight;   //top, bottom
                                //cloudProfile.volumeBox = float2(volumeBoxX,volumeBoxY)+cloudHeight-ViewPos.z*0.25f;   //top, bottom
        /*SHAPE*/                 cloudProfile.shape = float4(float3(volumeShapeX*CLOUD_COVERAGE,volumeShapeY*CLOUD_COVERAGE,volumeShapeZ*CLOUD_COVERAGE)+cloudHeight,volumeShapeThic);       //top, mid, bot, thickness
        /*SOFT*/                  cloudProfile.soft = softMul;
                                    //cloudProfile.shape = float4(float3(volumeShapeX*CLOUD_COVERAGE,volumeShapeY*CLOUD_COVERAGE,volumeShapeZ*CLOUD_COVERAGE)+cloudHeight-ViewPos.z*0.25f,volumeShapeThic);       //top, mid, bot, thickness
        /*BRIGHTNESS*/            cloudProfile.brightness = cloudBrightness;
        /*RANGE*/                 //cloudProfile.range = float3(0.9 + CLOUD_COVERAGE*0.16, 0.1, 0.2+CLOUD_COVERAGE*0.4),    //total, top, bottom
        /*RANGE*/                 cloudProfile.range = float3(rangeTotal, rangeTop, rangeBottom),    //total, top, bottom
        /*SOLIDNESS*/             cloudProfile._solidness = float2(solidness, solidnessBottom);   //top, bottom
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

        #ifdef SECONDARY_LAYER
        //Mid clouds
		/*MARCHING (ray tracing)*/ //{float4(clamp( (_cloudHeight-CameraPosition.z+_volumeShape.y*0.75f)/(_cloudHeight),1.0f,_minStep), _maxStep, _stepMult, _maxStepCount),   //min length, max length, multiplicand, max step
        /*MARCHING (ray tracing)*/ cloudProfile2.march = float4( minStep, maxStep, stepMult, maxStepCount);   //min length, max length, multiplicand, max step
        /*CUTOFF*/                 cloudProfile2.cutoff = float2(densCut, alphaCut);  // density cut, transparency cut
        /*VOLUME BOX*/            cloudProfile2.volumeBox = float2(300,0)+cloudHeight-50;   //top, bottom
                                //cloudProfile.volumeBox = float2(volumeBoxX,volumeBoxY)+cloudHeight-ViewPos.z*0.25f;   //top, bottom
        /*SHAPE*/                 cloudProfile2.shape = float4(float3(300,0,0)+cloudHeight-50,0.5);       //top, mid, bot, thickness
        /*SOFT*/                  cloudProfile2.soft = 0.5;
                                    //cloudProfile.shape = float4(float3(volumeShapeX*CLOUD_COVERAGE,volumeShapeY*CLOUD_COVERAGE,volumeShapeZ*CLOUD_COVERAGE)+cloudHeight-ViewPos.z*0.25f,volumeShapeThic);       //top, mid, bot, thickness
        /*BRIGHTNESS*/            cloudProfile2.brightness = cloudBrightness;
        /*RANGE*/                 //cloudProfile.range = float3(0.9 + CLOUD_COVERAGE*0.16, 0.1, 0.2+CLOUD_COVERAGE*0.4),    //total, top, bottom
        /*RANGE*/                 cloudProfile2.range = float3(1.0, 0.0, 1.0),    //total, top, bottom
        /*SOLIDNESS*/             cloudProfile2._solidness = float2(0.6, 0.2);   //top, bottom
        /*DENSITY CHUNK*/         cloudProfile2.densityChunk = float2(densCA*0.75, densCB*0.75);   //dens A,B
        /*SHADOW*/                cloudProfile2.shadow = float4(80, shadowDetailStrength*0.08, shadowExpand*0.08, shadowStrength*0.08*shadowMul);  //step length, detail strength, expanding, strength
        /*DISTORTION*/            cloudProfile2.distortion = float4(distMaxAngle, distStrength, distBumpStrength, distSmallBumpStrength);  //max angle, strength, bump strength, small bump strength
        /*FADE DISTANCE*/         cloudProfile2.fade = fadeDist;
        /*DETAIL DENSITY*/        cloudProfile2.densityDetail = float3(detailDensA,detailDensB,detailDensC);   //dens C,D,E
        /*CHUNK SCALE*/           cloudProfile2.scaleChunk = float3(scaleX*0.0001, scaleY*0.001, vertS);    //scale A,B, vertical stretch
        /*DETAIL SCALE*/          cloudProfile2.scaleDetail = float3(detailScaleA*0.001,detailScaleB*0.01,detailScaleC*0.01);   //scale C,D,E
        ///*CLOUD SHIFT*/            float3(-0.8, 0.0, 0.0),
        /*CLOUD SHIFT*/           cloudProfile2.cloudShift = float3(-0.085, 0.0, 0.0);
        /*NOISE OFFSET A*/        cloudProfile2.offsetA = nsOffsetA;
        /*NOISE OFFSET B*/        cloudProfile2.offsetB =  nsOffsetB;
        /*NOISE OFFSET C*/        cloudProfile2.offsetC =  nsOffsetC;
        /*NOISE OFFSET D*/        cloudProfile2.offsetD =  nsOffsetD;
        /*NOISE OFFSET E*/        cloudProfile2.offsetE =  nsOffsetE;
                                  cloudProfile2.startDistance = 0;
        #endif
                                  
}

float4 RenderClouds(float3 ViewDir, float3 ViewPos, float3 sunDir, float distance, float2 uv)
{


    updateProfiles();               
    CloudRandomness();
    CloudColorVarying(sunDir, baseColor);
    
    float3 lightDir = 0.0;
    float3 light = LightSource(sunDir, MoonDirection.xyz, lightDir);

    float4 clouds2 = float4(0.0, 0.0, 0.0, 0.0);
    float4 clouds = float4(0.0, 0.0, 0.0, 0.0);

    float4 cloud_temp;

    float3 bg_col = tex2Dlod(ReShade::BackBuffer,float4(uv,0,0)).xyz;

    float depth = distance;

    //if(ViewPos.z>cloudProfile.volumeBox.y+cloudHeight){
        clouds = CloudAtRay(bg_col,cloudProfile, baseColor, ViewDir, ViewPos, light, lightDir, timer*0.0001f, depth, uv);

        //clouds2 = CloudAtRay(bg_col.xyz+clouds.xyz*clouds.w,cloudProfile2, baseColor, ViewDir, ViewPos, light, lightDir, timer*0.0001f, depth, uv);
        //cloud_temp = clouds;
        //clouds2.rgb += cloud_temp.rgb*clouds2.w;
        //clouds2.w = clouds2.w*cloud_temp.w;
    /* }else{
        clouds = CloudAtRay(bg_col,cloudProfile2, baseColor, ViewDir, ViewPos, light, lightDir, timer*0.0001f, depth, uv);
        cloud_temp = CloudAtRay(bg_col,cloudProfile, baseColor, ViewDir, ViewPos, light, lightDir, timer*0.0001f, depth, uv);
        clouds.rgb += cloud_temp.rgb*clouds.w;
        //clouds.w = clouds.w*cloud_temp.w;
    } */

    
    /* float shad = 1.0;
    shad *= exp(-ShadowOnGround(cloudProfile,ViewPos+ViewDir*depth,lightDir));
    if(depth < 60000){
        shad = (1.0f-shad)*(saturate((1.0/(depth*0.001))));
        float term = clouds.w;
        clouds.xyz=lerp(clouds.xyz,bg_col*0.375f,saturate(shad*term));
        clouds.w-=shad*term;
    } */

    //clouds.rgb+=bg_col*clouds.w;
  
    return clouds;
}
float4 RenderCloudsWithRays(float3 ViewDir, float3 ViewPos, float3 sunDir, float distance, float2 uv)
{

   updateProfiles();

                                  


    CloudRandomness();
    CloudColorVarying(sunDir, baseColor);
    

    
    float3 lightDir = sunDir;
    float3 light = LightSource(sunDir, MoonDirection.xyz, lightDir);

    

    float4 cloud_temp = float4(0.0, 0.0, 0.0, 0.0);
    float4 clouds = float4(0.0,0.0,0.0,0.0);
    float4 cloudGodRays = float4(0.0,0.0,0.0,0.0);

        float3 bg_col = tex2Dlod(ReShade::BackBuffer,float4(uv,0,0)).xyz;

        float depth = distance;

        clouds = CloudsWithRays(bg_col,cloudProfile, baseColor, normalize(ViewDir), ViewPos, light, normalize(lightDir), timer*0.0001f, depth, uv);
    
    /* float shad = 1.0;
    shad *= exp(-ShadowOnGround(cloudProfile,ViewPos+ViewDir*depth,normalize(lightDir)));
    if(depth < 60000){
        shad = (1.0f-shad)*(saturate((1.0/(depth*0.001))));
        float term = clouds.w;
        clouds.xyz=lerp(clouds.xyz,bg_col*0.375f,saturate(shad*term));
        clouds.w-=shad*term;
    } */
  
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

static const int bayerFilter[16] = {
    0,8,2,10,
    12,4,14,6,
    3,11,1,9,
    15,7,13,5
};

bool writePixel(float2 uv, int frame){
    int2 coord = int2(uv);
    int id = frame % 16;
    return (((coord.x + 4 * coord.y) % 16) == bayerFilter[id]);
}

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
    //if(!writePixel(tex*resolution,framecount)){discard;}


	float2 fuv = -uv;

    float4 cl = 0.0f;

    float2 texel = 1.0 / resolution;


    float3 worldPos = DepthToWorldPos(CustomDepth(uv),uv);
    float dp = length(camPos-worldPos);

    float depth = dp;



    
    /* float depthTest = 0.0;
    for(int i = 0;i<9;i++){
        depthTest = dp;
        if(depthTest < depth){
            depth = depthTest;
        }
    } */

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
        depthTest = length(camPos-DepthToWorldPos(CustomDepth(tex + texel * gaussKernel3x3[i].xy * 2.0),tex)).x;
        if(depthTest < depth){
            depth = depthTest;
        }
    }

    if(length(depth-fillDepth) >= (depthFill*2.0f)*depth*4.0f && fillDepth < 125000){
        return float4(1,0,0,1);
    }
    return float4(0,0,0,0);
}
float PS_EdgeMaskGrow(float4 vpos : SV_Position, float2 tex : TexCoord) : SV_Target
{   
    float2 texel = 1.0 / float2(BUFFER_WIDTH, BUFFER_HEIGHT);

    float mask = tex2D(RTSamplerFullMask,tex.xy).x;

       for(int i=1;i<=2; i++)
    {
        for(int j=0;j<8;j++){
            mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * gaussKernel3x3[j].xy*i).x);
        }
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

     for(int i=1;i<=2; i++)
    {
        for(int j=0;j<8;j++){
            mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * gaussKernel3x3[j].xy*i).x);
        }
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

    for(int i=1;i<=2; i++)
    {
        for(int j=0;j<8;j++){
            mask = max(mask, tex2D(RTSamplerFullMask,tex.xy + texel * gaussKernel3x3[j].xy*i).x);
        }
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
            _texture = smartDeNoise(RTSampler512,res,uv,1.5,1.5,0.025);
        }else if(deNoiseLevel == 2){
            _texture = smartDeNoise(RTSampler512,res,uv,2.0,2.0,0.05);
        }else if(deNoiseLevel == 3){
            _texture = smartDeNoise(RTSampler512,res,uv,2.0,2.0,0.1);
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
            _texture = smartDeNoise(RTSampler1024,res,uv,1.5,1.5,0.025);
        }else if(deNoiseLevel == 2){
            _texture = smartDeNoise(RTSampler1024,res,uv,2.0,2.0,0.05);
        }else if(deNoiseLevel == 3){
            _texture = smartDeNoise(RTSampler1024,res,uv,2.0,2.0,0.1);
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
            _texture = smartDeNoise(RTSamplerFull,res,uv,1.5,1.5,0.025);
        }else if(deNoiseLevel == 2){
            _texture = smartDeNoise(RTSamplerFull,res,uv,2.0,2.0,0.05);
        }else if(deNoiseLevel == 3){
            _texture = smartDeNoise(RTSamplerFull,res,uv,2.0,2.0,0.1);
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
    
    if(length(col/3)<0.00001f+cableShit*0.0001f && depth > 0.95f && GameTime.x > 4 && GameTime.x < 21){
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

/*
float4 LinearFog(float2 uv)
{
    float d = GetLinearDepth(uv);

    return float4(0.0,0.0,0.0,0.0);
}
*/

float4 PS_Final(float4 vpos : SV_Position, float2 uv :TexCoord) : SV_Target
{
    float4 col = tex2Dlod(ReShade::BackBuffer, float4(uv,0.0,0.0));
    if(CLOUD_COVERAGE <= 0.019){return col;}
    updateProfiles();
    CloudRandomness();
    //float4 cl = PS_CloudsCA(RTSamplerFull2,uv);
    float4 cl = tex2D(RTSamplerFull2,uv);

    
    float3 camPos = float3(InverseView[3].x, InverseView[3].y, InverseView[3].z);
    float3 worldPos = DepthToWorldPos(CustomDepth(uv),uv);
    float3 rayDir = normalize(worldPos - camPos)*-1;
    float depth = length(camPos-worldPos);
    /*
    float3 lightDir;
    LightSource(SunDirection,MoonDirection,lightDir);
    float shad = 1.0;
    shad *= exp(-ShadowOnGround(cloudProfile,camPos+rayDir*depth,lightDir)*0.1);
    if(depth < 60000){
        shad = (1.0f-shad)*(saturate((1.0/(depth*0.001))));

        float3 c_s = float3(0.125,0.164,0.2);
        float3 c_s_d = float3(c_s.x-col.x,c_s.y-col.y,c_s.z-col.z);
        float3 d_c = lerp(col.xyz,col.xyz*c_s,0.75);

        col.xyz = lerp(col.xyz,col.xyz*d_c,saturate(shad*1.25));
    }
    */
    /* float d = clampMap(GetLinearDepth(uv),0,1,FOG_Min,FOG_Max);
    float ci = exp(-ShadowOnGround(cloudProfile,camPos+rayDir*depth,0));
    float3 lfc = lerp(ColorAtSun,ColorBase,ci*0.8);
    //col.xyz = lerp(col.xyz,lfc,d*ci);
    //float mask = tex2D(RTSamplerFullMask,uv).x; */
    if(length(cl.xyz) > 0){

		cl.xyz = cl.xyz+col.xyz*saturate(cl.w*AlphaBlend);
		col.xyz = lerp(col.xyz,cl.xyz,saturate(1.0f-cl.w));
	}


    //if(mask!=0){col.xyz=float3(1.0,0,0);}
    //if(uv.x>0.7){col.xyz = CustomDepth(uv);}
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
