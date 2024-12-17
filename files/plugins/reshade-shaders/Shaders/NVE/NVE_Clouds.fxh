#include "ReShadeUI.fxh"

#define hide true
#define hideDebug true

uniform float4x4 InverseView < hidden = hideDebug; >;
uniform float4x4 WorldView < hidden = hideDebug; >;
uniform float4x4 WorldInverseViewProjection < hidden = hideDebug; >;
uniform float4x4 WorldViewProjection < hidden = hideDebug; >;
uniform float4x4 Projection < hidden = hideDebug; >;

uniform float3 AzimuthWestColor < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 AzimuthEastColor < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 AzimuthTransitionColor < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 ZenithColor < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 ZenithTransitionColor < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float4 SkyPlaneColor < hidden = hideDebug; > = float4(0.0f, 0.0f, 0.0f, 0.0f);

uniform float3 SunColor < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 SunColorHdr < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 SunDiscColorHdr < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 SmallCloudColorHdr < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 MoonColor < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);

uniform float3 CloudBaseMinusMidColor < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 CloudMidColor < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 CloudShadowMinusBaseColorTimesShadowStrength < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);

uniform float3 SunPosition < hidden = hideDebug; >;
uniform float3 SunDirection < hidden = hideDebug; >;
uniform float3 MoonPosition < hidden = hideDebug; >;
uniform float3 MoonDirection < hidden = hideDebug; >;
uniform float3 GameTime < hidden = hideDebug; >;
uniform float3 WeatherData < hidden = hideDebug; >;
uniform float3 WindData < hidden = hideDebug; >;
uniform float3 NOffsetA < hidden = hideDebug; >;
uniform float3 NOffsetB < hidden = hideDebug; >;
uniform float3 cloudShift < hidden = hideDebug; >;
uniform float3 nsOffsetA < hidden = hideDebug; >;
uniform float3 nsOffsetB < hidden = hideDebug; >;
uniform float3 nsOffsetC < hidden = hideDebug; >;
uniform float3 nsOffsetD < hidden = hideDebug; >;
uniform float3 nsOffsetE < hidden = hideDebug; >;
uniform float3 clampBright <    ui_type = "color";ui_min = 0.0f;ui_max = 1.0f; hidden=hideDebug;> = float3(0.0f,0.0f,0.0f);
uniform float3 ColorAtSun <     ui_type = "color";ui_min = 0.0f;ui_max = 1.0f; hidden=hideDebug;> = float3(0.0f,0.0f,0.0f);
uniform float3 ColorBase <      ui_type = "color";ui_min = 0.0f;ui_max = 1.0f; hidden=hideDebug;> = float3(0.0f,0.0f,0.0f);


uniform float4x4 cs1 < hidden = hideDebug; >;
uniform float4x4 cs2 < hidden = hideDebug; >;
uniform float4x4 cs3 < hidden = hideDebug; >;
uniform float4x4 cs4 < hidden = hideDebug; >;
uniform float4x4 cs5 < hidden = hideDebug; >;
uniform float4x4 cs6 < hidden = hideDebug; >;
uniform float4x4 cs7 < hidden = hideDebug; >;
uniform float4x4 cs8 < hidden = hideDebug; >;
uniform float4x4 cs9 < hidden = hideDebug; >;

uniform float oof <hidden = hideDebug;> = 0.0f;

#define InScatterExp (cs4[2][2])
#define InScatterMul (cs4[2][3])

#define L_Probability (cs7[1][1])
#define L_Frequency (cs7[1][3])
#define L_StrobeSpeed (cs7[2][3])
#define L_StrobeMul (cs7[3][3])
#define L_Count (cs8[3][2])
#define L_Curve (cs9[1][2])
#define L_Size (cs9[1][3])

#define L_CenterX (cs9[0][0])
#define L_CenterY (cs9[0][1])
#define L_CenterZ (cs9[0][2])

#define BottomDetailMul (cs9[0][3])
#define BottomDetailLow (cs9[1][0])
#define BottomDetailHigh (cs9[1][1])

#define PowderExp (cs6[2][1])
#define PowderStrength (cs6[2][0])

//uniform bool SEPARATOR0 < __UNIFORM_INPUT_BOOL1 ui_label = "Density and Coverage"; > = false;
#define coverageBottom (cs1[0][1])
#define cloudMapStrength (cs1[0][2])
#define cloudmapScale (cs2[0][1])
#define cloudmapZScale (cs2[0][2])
#define cloudmapOffset (cs2[0][3])
#define detailDensA (cs1[0][3])
#define detailDensB (cs1[1][0])
#define altitudeOffsetB (cs3[3][1])
#define densExp (cs6[2][2])
#define detailDensC (cs1[1][1])
#define C_Smooth (cs6[2][3])
#define C_Smooth2 (cs6[3][0])
#define C_Smooth3 (cs6[3][1])
#define C_Contrast (cs6[3][2])
#define densExp2 (cs6[3][3])
#define detailScaleA (cs1[1][2])
#define detailScaleB (cs1[1][3])
#define detailScaleC (cs1[2][0])
#define detailZScaleA (cs3[2][2])
#define detailZScaleB (cs3[2][3])
#define detailZScaleC (cs3[3][0])

uniform bool SEPARATOR11 < __UNIFORM_INPUT_BOOL1 ui_label = "Scale"; hidden = hide; > = false;
#define densCA (cs3[0][2])
#define densCB (cs3[0][3])
#define scaleX (cs3[1][2])
#define scaleY (cs3[1][3])
#define vertS (cs3[2][0])
#define chunkBZScale (cs3[2][1])

uniform bool SEPARATOR1 < __UNIFORM_INPUT_BOOL1 ui_label = "Steps Configuration"; hidden = hide; > = false;
#define maxStepCount (int(cs1[2][3]))
#define stepMult (int(cs1[3][0]))
#define minStep (int(cs1[3][1]))
#define maxStep (int(cs1[3][2]))
#define mainNoise (cs5[2][1])
#define alphaCut (cs1[3][3])
#define densCut (cs5[1][3])
#define fadeDist (cs2[0][0])
#define dfLimit (cs5[2][0])
#define RMMask1 (cs5[2][3])
#define RMMask2 (cs5[3][0])
#define MaskDistOffset (cs5[3][1])

uniform bool SEPARATOR2 < __UNIFORM_INPUT_BOOL1 ui_label = "Volume Configuration"; hidden = hide; > = false;
#define cloudHeight (cs2[1][0])
#define volumeBoxX (cs2[1][1])
#define volumeBoxY (cs2[1][2])
#define volumeShapeX (cs2[1][3])
#define volumeShapeY (cs2[2][0])
#define volumeShapeZ (cs2[2][1])
#define volumeShapeThic (cs2[2][2])
#define rangeTotal (cs7[0][0])
#define rangeBottom (cs7[0][1])
#define rangeTop (cs7[0][2])
#define softMul (cs7[0][3])
#define solidness (cs3[1][0])
#define solidnessBottom (cs3[1][1])

uniform bool SEPARATOR3 < __UNIFORM_INPUT_BOOL1 ui_label = "Shadow Configuration"; hidden = hide; > = false;
#define shadowStepLength (cs2[2][3])
#define shadowSteps (cs7[1][0])
#define shadowNoise (cs5[2][2])
#define shadowExpand (cs2[3][0])
#define shadowStrength (cs2[3][1])
#define shadowDetailStrength (cs2[3][2])
#define shadowThreshold (cs7[1][2])
#define shadowLimit (cs2[3][3])
#define shadowEarlyExit (cs3[0][0])
#define shadowEarlyExitApprox (cs3[0][1])

uniform bool SEPARATOR4 < __UNIFORM_INPUT_BOOL1 ui_label = "Distortion Configuration"; hidden = hide; > = false;
#define distMaxAngle (cs3[3][2])
#define distStrength (cs3[3][3])
#define distBumpStrength (cs4[0][0])
#define distSmallBumpStrength (cs4[0][1])

uniform bool SEPARATOR41 < __UNIFORM_INPUT_BOOL1 ui_label = "Earth Configuration"; hidden = hide; > = false;
#define earthShadVal1 (cs4[1][3])
#define earthShadVal2 (cs4[2][0])
#define earthShadVal3 (cs4[2][1])
#define earthShadVal4 (cs4[0][3])
#define earthShadVal5 (cs4[1][0])
#define earthShadVal6 (cs4[1][1])
#define earthShadVal7 (cs4[1][2])
#define cBallRad (cs6[1][1])
#define cBallClipBegin (cs6[1][2])
#define cBallClipEnd (cs6[1][3])
#define atmoConvDist (cs6[0][2])
#define atmDensityConfig (cs6[0][3])
uniform float atmHeight <    ui_type = "slider";ui_min = 0.0f;ui_max = 100000.0f; hidden=hide;> = 0.6f;

uniform bool SEPARATOR5 < __UNIFORM_INPUT_BOOL1 ui_label = "Blending Configuration"; hidden = hide; > = false;
#define atmoDistance2 (cs4[2][2])
#define atmoMaxBlend2 (cs4[2][3])
#define AtmoDistanceExp (cs7[1][3])
#define edgeSmooth (cs4[0][2])
#define sunScale (cs4[3][2])
#define sunScaleExp (cs4[3][3])
#define depthMul (cs5[0][3])
uniform float depthFill <           ui_type = "slider";ui_min = 0.0f;ui_max = 1.0f; hidden=hide;> = 1.0f;
#define borderRadius (cs6[0][0])
#define borderRadius2 (cs6[0][1])

uniform bool SEPARATOR6 < __UNIFORM_INPUT_BOOL1 ui_label = "Shading Configuration"; hidden = hide; > = false;
uniform float3 clampBrightConfig <    ui_type = "color";ui_min = 0.0f;ui_max = 1.0f; hidden=hide;> = float3(0.0f,0.0f,0.0f);
uniform float3 ColorAtSunConfig <     ui_type = "color";ui_min = 0.0f;ui_max = 1.0f; hidden=hide;> = float3(0.0f,0.0f,0.0f);
uniform float3 ColorBaseConfig <      ui_type = "color";ui_min = 0.0f;ui_max = 1.0f; hidden=hide;> = float3(0.0f,0.0f,0.0f);
#define cloudBrightness (cs4[3][1])

#define sunStrength (cs7[2][0])
#define mieStrengthConfig (cs1[2][1])

#define HGStrength (cs5[0][0])
#define HGStrengthTop (cs5[0][1])
#define HGMu (cs5[0][2])

#define blur (cs5[1][0])
#define timeBoost (cs5[1][1])
#define timeSkip (cs5[1][2])

uniform int usePowder <             ui_type = "slider";ui_min = 0;ui_max = 2; hidden=hide;> = 1;
uniform float pwd <                 ui_type = "slider";ui_min = 0.0f;ui_max = 10.0f; hidden=hide;> = 0.0f;
uniform float pwd2 <                ui_type = "slider";ui_min = 0.0f;ui_max = 1.0f; hidden=hide;> = 0.0f;

uniform bool SEPARATOR7 < __UNIFORM_INPUT_BOOL1 ui_label = "Other Configuration"; hidden = hide; > = false;
#define deNoiseLevel (cs5[3][2])
#define cloudsAberration (cs5[3][3])
#define cloudsSharpen (cs1[2][2])

#define cableShit (cs6[1][0])
#define AlphaBlend (cs7[2][1])

#define shadowMul (cs7[2][2])

#define atmoStrength (cs7[3][0])
#define rayDensity (cs7[3][1])
#define boxOffset (cs7[3][2])

#define groundAtmoStartDistance (cs8[0][0])
#define groundAtmoBlendClose (cs8[0][1])
#define groundAtmoBlendFar (cs8[0][2])
#define groundAtmoBlendCurve (cs8[0][3])
#define groundAtmoBlendMax (cs8[1][0])
#define skyAtmoStartDistance (cs8[1][1])
#define skyAtmoBlendClose (cs8[1][2])
#define skyAtmoBlendFar (cs8[1][3])
#define skyAtmoBlendCurve (cs8[2][0])
#define skyAtmoBlendMax (cs8[2][1])
#define shadowOffset (cs8[2][2])
#define csBottomOffset (cs8[2][3])
#define atmoRaySize (cs8[3][0])
#define brightProtect (cs8[3][1])

uniform float3 AmbientSkyColor < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);

#define ambient_factor (cs9[2][0])
#define ambient_power (cs9[2][1])
#define ambient_altitude_bottom (cs9[2][2])
#define ambient_altitude_top (cs9[2][3])
#define ray_shape_offset (cs9[3][0])
#define ray_altitude_begin (cs9[3][1])
#define ray_altitude_end (cs9[3][2])
#define ray_shape (cs9[3][3])

uniform float3 tornado_pos <hidden=hide;> = float3(0.0f, 0.0f, 0.0f);
uniform float tornado_radius <hidden=hide;> = float3(0.0f, 0.0f, 0.0f);


uniform float C_fadeDist <				ui_type = "slider";ui_min = 0.0f;ui_max = 1.0f; hidden=hide;> = 1.0f;

uniform float3 SkyCol <				ui_type = "color";ui_min = 0.0f;ui_max = 1.0f; hidden=hide;> = float3(0.0f,0.0f,0.0f);
uniform float AtmoBleedStart <		ui_type = "slider";ui_min = 0.0f;ui_max = 2000.0f; hidden=hide;> = 0.0f;
uniform float AtmoBleedEnd <		ui_type = "slider";ui_min = 0.0f;ui_max = 5000.0f; hidden=hide;> = 1000.0f;
uniform float fakeAtmo <		ui_type = "slider";ui_min = 0.0f;ui_max = 1.0f; hidden=hide;> = 0.1f;
uniform float offsetBoxStart <		ui_type = "slider";ui_min = -2000.0f;ui_max = 0.0f; hidden=hide;> = 0.0f;
uniform float RMRaysMask1 <             ui_type = "slider";ui_min = 0.0f;ui_max = 2.0f; hidden=hide;> = 0.6f;
uniform float RMRaysMask2 <             ui_type = "slider";ui_min = 0.0f;ui_max = 2.0f; hidden=hide;> = 0.6f;
uniform int RaysmaxStepCount <          ui_type = "slider";ui_min = 0;ui_max = 512; hidden=hide;> = 512;
uniform int RaysstepMult <              ui_type = "slider";ui_min = 0;ui_max = 25; hidden=hide;> = 12;
uniform int RaysminStep <               ui_type = "slider";ui_min = 0;ui_max = 25; hidden=hide;> = 8;
uniform int RaysmaxStep <               ui_type = "slider";ui_min = 0;ui_max = 25; hidden=hide;> = 12;
uniform float RaysmainNoise <           ui_type = "slider";ui_min = 0.0f;ui_max = 10.0f; hidden=hide;> = 0.6f;
uniform float RaysalphaCut <            ui_type = "slider";ui_min = 0.0f;ui_max = 0.06f; hidden=hide;> = 0.04f;
uniform float RaysdensCut <             ui_type = "slider";ui_min = 0.0f;ui_max = 1.0f; hidden=hide;> = 0.6f;
uniform float RaysfadeDist <            ui_type = "slider";ui_min = 0.0f;ui_max = 100000.0f; hidden=hide;> = 60000.0f;

uniform float3 AmbientEast < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float3 AmbientWest < hidden = hideDebug; > = float3(0.0f, 0.0f, 0.0f);
uniform float4 AmbientExtra < hidden = hideDebug; > = float4(0.0f, 0.0f, 0.0f, 0.0f);
#define ambientSplit (AmbientExtra.x)
#define ambientMul (AmbientExtra.y)