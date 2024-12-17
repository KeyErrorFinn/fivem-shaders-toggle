/*-------------------------------
	ENBFeeder for ReShade
	(c) Quant 2020-2023

https://www.gtainside.com/en/gta5/mods/143054-enbfeeder/
--------------------------------*/


//EInteriorFactor (fixed), 0 = exterior, 1 = interior.
uniform float ExteriorInterior < hidden = true; > ;

//vertical fov in degrees.
uniform float FieldOfView < hidden = true; > ;

//time in 0..24 format.
uniform float GameTime < hidden = true; > ;

//x = current weather, y = incoming weather, z = transition (0..1), w = region factor (0 = Global, 1 = Urban). Weather index: 0 = not found, 1 = extrasunny, 2 = clear, 3 = clearing, 4 = clouds, 5 = overcast, 6 = smog, 7 = foggy, 8 = rain, 9 = thunder, 10 = blizzard, 11 = neutral, 12 = snow, 13 = snowlight, 14 = xmas, 15 = halloween.
uniform float4 qWeather < hidden = true; > ;

//x,y = player coords in screen, z = distance from camera, w = 1 when player is in screen, 0 when is on back side.
uniform float4 PlayerPos < hidden = true; > ;

//x,y = wind direction, z = speed.
uniform float3 Wind < hidden = true; > ;

//similar to enbeffect's dofProj.
uniform float4 DofProj < hidden = true; > ;

//sun in world space.
uniform float3 sunDirection < hidden = true; > ;
uniform float3 sunPosition < hidden = true; > ;

//sun in screen space. x,y = 2D position, z = 1 when sun is in front of camera, 0 when is on back side.
uniform float3 SunScreenPos < hidden = true; > ;

//moon in world space.
uniform float3 moonDirection < hidden = true; > ;
uniform float3 moonPosition < hidden = true; > ;

//moon in screen space. x,y = 2D position, z = 1 when moon is in front of camera, 0 when is on back side.
uniform float3 MoonScreenPos < hidden = true; > ;

//World View Matrix Transforms
uniform float4 WorldView0 < hidden = true; > ;
uniform float4 WorldView1 < hidden = true; > ;
uniform float4 WorldView2 < hidden = true; > ;
uniform float4 WorldView3 < hidden = true; > ;
#define WorldView float4x4(WorldView0, WorldView1, WorldView2, WorldView3)

uniform float4 WorldViewProj0 < hidden = true; > ;
uniform float4 WorldViewProj1 < hidden = true; > ;
uniform float4 WorldViewProj2 < hidden = true; > ;
uniform float4 WorldViewProj3 < hidden = true; > ;
#define WorldViewProj float4x4(WorldViewProj0, WorldViewProj1, WorldViewProj2, WorldViewProj3)

uniform float4 ViewInverse0 < hidden = true; > ;
uniform float4 ViewInverse1 < hidden = true; > ;
uniform float4 ViewInverse2 < hidden = true; > ;
uniform float4 ViewInverse3 < hidden = true; > ;
#define ViewInverse float4x4(ViewInverse0, ViewInverse1, ViewInverse2, ViewInverse3)

uniform float4 WorldInverseVP0 < hidden = true; > ;
uniform float4 WorldInverseVP1 < hidden = true; > ;
uniform float4 WorldInverseVP2 < hidden = true; > ;
uniform float4 WorldInverseVP3 < hidden = true; > ;
#define WorldInverseVP float4x4(WorldInverseVP0, WorldInverseVP1, WorldInverseVP2, WorldInverseVP3)

#define CameraPosition ViewInverse3
