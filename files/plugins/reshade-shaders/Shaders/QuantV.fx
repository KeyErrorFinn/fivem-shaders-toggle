//QuantV ReShade effect

uniform float Corona_Light_Intensity < ui_label = "Intensity (Global)"; ui_category = "CORONA LIGHTS"; ui_type = "drag"; ui_min = 0.01; ui_max = 500.0; > = 1.0;
uniform float Corona_Light_Emergency_Intensity < ui_label = "Intensity (Emergency Vehicles)"; ui_category = "CORONA LIGHTS"; ui_type = "drag"; ui_min = 0.01; ui_max = 500.0; > = 1.5;
uniform int Corona_Light_Style < ui_label = "Style"; ui_category = "CORONA LIGHTS"; ui_type = "combo"; ui_items = "Smooth Rays\0Rays V3\0Thin Rays\0"; > = 0;
uniform bool Disable_Corona_Light_from_Vehicles < ui_label = "Disable from Vehicles"; ui_category = "CORONA LIGHTS"; > = false;
uniform bool Disable_Corona_Light_Everywhere < ui_label = "Disable Everywhere"; ui_category = "CORONA LIGHTS"; > = false;
uniform int sep0 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool Streetlights < ui_label = "Use Streetlights"; ui_category = "STREETLIGHTS"; > = false;
uniform float Streetlights_Mult < ui_label = "Intensity"; ui_category = "STREETLIGHTS"; ui_type = "drag"; ui_min = 0.01; ui_max = 10.0; > = 1.0;
uniform float3 Streetlights_Col < ui_label = "Color"; ui_category = "STREETLIGHTS"; ui_type = "color"; > = float3(1.0, 0.41, 0.0);
uniform float Streetlights_Reflect < ui_label = "Reflect Intensity"; ui_category = "STREETLIGHTS"; ui_type = "drag"; ui_min = 0.01; ui_max = 100.0; > = 1.0;
uniform int sep1 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool Screen_Raindrops_Classic < ui_label = "Raindrops: Classic"; ui_category = "SCREEN RAINDROPS"; > = false;
uniform bool Screen_Raindrops_Racing < ui_label = "Raindrops: Racing Style"; ui_category = "SCREEN RAINDROPS"; > = true;
uniform bool Screen_Raindrops_NFS < ui_label = "Raindrops: NFS Style"; ui_category = "SCREEN RAINDROPS"; > = false;
uniform bool Screen_Raindrops_Force_Enable < ui_label = " > Force Enable"; ui_category = "SCREEN RAINDROPS"; > = false;
uniform float Screen_Raindrops_Scale < ui_label = "Scale"; ui_category = "SCREEN RAINDROPS"; ui_tooltip = "Scale size of the droplets"; ui_type = "drag"; ui_min = 0.25; ui_max = 10.0; > = 1.0;
uniform bool Screen_Frozen_Snow < ui_label = "Screen Frozen Snow"; ui_spacing = 2; ui_category = "SCREEN RAINDROPS"; > = false;
uniform bool Screen_Frozen_Snow_ForceEnable < ui_label = " > Force Enable"; ui_category = "SCREEN RAINDROPS"; > = false;
uniform int sep2 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform float2 vehSpecIntensity < ui_label = "Specular Intensity"; ui_tooltip = "DAY    |    NIGHT    (a.k.a Reflection Amount)"; ui_category = "VEHICLE"; ui_type = "drag"; > = float2(1.1, 1.2);
uniform float2 vehFresnel < ui_label = "Specular Fresnel"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE"; ui_type = "drag"; > = float2(0.98, 0.95);
uniform float2 vehFalloff < ui_label = "Specular Falloff"; ui_tooltip = "DAY    |    NIGHT    (a.k.a Reflection Glossiness)"; ui_category = "VEHICLE"; ui_type = "drag"; > = float2(1.0, 1.0);
uniform float2 vehMask < ui_label = "Specular Mask"; ui_tooltip = "DAY    |    NIGHT    (a.k.a Metallic Amount)"; ui_category = "VEHICLE"; ui_type = "drag"; > = float2(1.3, 1.5);
uniform bool vehSSR < ui_label = "Vehicle SSR"; ui_category = "VEHICLE"; ui_tooltip = "Vehicle - Screen Space Reflections"; ui_spacing = 2; > = true;
uniform int Vehicle_Paint < ui_label = "Paint Shader"; ui_category = "VEHICLE"; ui_type = "combo"; ui_items = "Default Paint\0Metallic Paint (Pearlescent)\0Iridescent (Metal)\0Iridescence Wrap (Pearlescent/Matte/Metal)\0Carbon Fiber (Pearlescent/Matte)\0Opal (Pearlescent/Matte/Metal)\0Brushed Metal (Pearlescent/Metal/Chrome)\0Colored Chrome (Chrome)\0"; ui_spacing = 2; > = 0;
uniform bool Vehicle_Carbon < ui_label = "Custom Carbon Fiber"; ui_category = "VEHICLE"; ui_spacing = 2; > = true;
uniform bool Use_Colored_Carbon_Fiber < ui_label = "Use Colored Carbon Fiber"; ui_category = "VEHICLE"; > = false;
uniform float3 Color_Carbon_Fiber < ui_label = "Colored Carbon Fiber"; ui_type = "color"; ui_category = "VEHICLE"; > = float3(0.01, 0.01, 0.01);
uniform float2 vehGlass < ui_label = "Glass Reflection"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE"; ui_type = "drag"; ui_spacing = 2; > = float2(1.9, 2.5);
uniform float vehGlassTint < ui_label = "Glass Tint Opacity"; ui_category = "VEHICLE"; ui_type = "drag"; ui_min = 0.01; ui_max = 10.0; > = 1.0;
uniform float2 Vehicle_Chrome_Reflection < ui_label = "Chrome Reflection"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE"; ui_type = "drag"; ui_spacing = 2; > = float2(1.1, 1.4);
uniform float2 Vehicle_Chrome_Specular < ui_label = "Chrome Specular"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE"; ui_type = "drag"; > = float2(1.0, 1.0);
uniform float2 Vehicle_Wheel_Specular < ui_label = "Wheels Specular"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE"; ui_type = "drag"; ui_spacing = 2; > = float2(1.0, 1.0);
uniform float2 Vehicle_Rims_Reflection < ui_label = "Rims Reflection"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE"; ui_type = "drag"; > = float2(1.05, 1.3);
uniform float2 Vehicle_Tire_Reflection < ui_label = "Tire Reflection"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE"; ui_type = "drag"; > = float2(1.0, 1.0);
uniform float2 Vehicle_Tire_Wetness < ui_label = "Tire Wetness"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE"; ui_type = "drag"; > = float2(1.1, 1.1);
uniform bool Vehicle_Wheel_Chrome_Color < ui_label = "Wheel Chrome Color (pearlescent)"; ui_category = "VEHICLE"; ui_tooltip = "Paint 3. Pearlescent Color"; > = false;
uniform int Vehicle_Wheel_HotDisc < ui_label = "Wheel Brake Disc"; ui_category = "VEHICLE"; ui_type = "combo"; ui_items = "default\0Hot\0Very Hot\0"; ui_tooltip = "(Player must be inside the vehicle)"; > = 0;
uniform bool Vehicle_Raindrops < ui_label = "Vehicle Raindrops"; ui_category = "VEHICLE"; ui_tooltip = "Vehicle Rain Drops effect V2"; ui_spacing = 2; > = true;
uniform bool Vehicle_Rain_ForceEnable < ui_label = " > Force Enable"; ui_category = "VEHICLE"; > = false;
uniform bool Vehicle_Lower_Snow < ui_label = "Vehicle Lower Snow"; ui_category = "VEHICLE"; ui_spacing = 2;> = true;
uniform bool Vehicle_Top_Snow < ui_label = "Vehicle Top Snow"; ui_category = "VEHICLE"; > = false;
uniform bool Vehicle_Snow_Force < ui_label = " > Force Enable"; ui_category = "VEHICLE"; ui_tooltip = "Make sure your vehicle isn't 100% clean ;)"; > = false;
uniform int sep3 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool Vehicle_Glass_Refraction < ui_label = "GlassLight Refraction effect"; ui_category = "VEHICLE LIGHTS"; > = true;
uniform int Vehicle_Emergency_Lights < ui_label = "Emergency Lights Preset"; ui_category = "VEHICLE LIGHTS"; ui_type = "combo"; ui_items = "default\0Bright\0Very Bright\0"; ui_spacing = 2; > = 0;
uniform float2 Emissive_Global < ui_label = "Emissive Global"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE LIGHTS"; ui_type = "drag"; ui_min = 0.01; ui_max = 300.0; ui_spacing = 2; > = float2(1.3, 1.3);
uniform float2 Emissive_Emergency < ui_label = "Emissive Emergency"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE LIGHTS"; ui_type = "drag"; ui_min = 0.01; ui_max = 300.0; > = float2(1.9, 1.9);
uniform float2 Emissive_Headlight < ui_label = "Emissive Headlight"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE LIGHTS"; ui_type = "drag"; ui_min = 0.01; ui_max = 300.0; > = float2(1.0, 1.0);
uniform float2 Emissive_Taillight < ui_label = "Emissive Taillight"; ui_tooltip = "DAY    |    NIGHT"; ui_category = "VEHICLE LIGHTS"; ui_type = "drag"; ui_min = 0.01; ui_max = 300.0; > = float2(1.0, 1.0);
uniform float Emissive_Indicator < ui_label = "Emissive Indicator"; ui_category = "VEHICLE LIGHTS"; ui_type = "drag"; ui_min = 0.01; ui_max = 300.0; > = 1.0;
uniform float4 Vehicle_Headlight_Tint < ui_label = "Headlight Tint"; ui_category = "VEHICLE LIGHTS"; ui_type = "color"; ui_spacing = 2; > = float4(1.0, 1.0, 1.0, 0.0);
uniform float4 Vehicle_Taillight_Tint < ui_label = "Taillight Tint"; ui_category = "VEHICLE LIGHTS"; ui_type = "color"; > = float4(1.0, 0.2, 0.0, 0.0);
uniform float Emergency_Reflect < ui_label = "Emergency Reflect Intensity"; ui_category = "VEHICLE LIGHTS"; ui_type = "drag"; ui_min = 0.01; ui_max = 30.0; ui_spacing = 2; > = 1.0;
uniform float Emergency_Reflect_Far < ui_label = "Emergency Reflect Distance"; ui_category = "VEHICLE LIGHTS"; ui_type = "slider"; ui_min = 0.01; ui_max = 2.0; > = 1.0;
uniform bool Vehicle_Indicator_Animation < ui_label = "Indicator Led Animation"; ui_category = "VEHICLE LIGHTS"; ui_spacing = 2; > = false;
uniform int Animation_Profile < ui_label = "Indicator Animation Profile"; ui_tooltip = "if default animation doesn't match vehicle's animation then switch to the correct profile"; ui_category = "VEHICLE LIGHTS"; ui_type = "combo"; ui_items = "Default (auto)\0Profile#1\0Profile#2\0Profile#3\0Profile#4\0Profile#5\0Profile#6\0Profile#B1\0Profile#B2\0Profile#B3\0"; > = 0;
uniform bool Vehicle_DRL_Indicator < ui_label = "DRL Indicator"; ui_tooltip = "Indicator - Daytime Running Lights"; ui_category = "VEHICLE LIGHTS"; > = false;
uniform bool Vehicle_Lights_Flicker < ui_label = "Lights Flicker"; ui_category = "VEHICLE LIGHTS"; ui_tooltip = "Stroboscopic effect - Vehicle lights flickering in the camera for cinematic effect"; ui_spacing = 2; > = false;
uniform int sep4 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool ParticleFX < ui_category = "PARTICLE FX"; > = true;
uniform int Fire_Exhaust < ui_type = "combo"; ui_label = "Exhaust Fire Color"; ui_items = "Default\0Blue\0Red\0Fire\0Green\0RGB\0"; ui_category = "PARTICLE FX"; ui_tooltip = "Exhaust Fire / Nitrous"; > = 3;
uniform float Vehicle_Exhaust_Fire < ui_label = "Exhaust Fire Intensity"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PARTICLE FX"; ui_tooltip = "Exhaust Fire / Nitrous"; > = 1.0;
uniform float Vehicle_Exhaust_Fire_Scale < ui_label = "Exhaust Fire Scale"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PARTICLE FX"; ui_tooltip = "Exhaust Fire / Nitrous"; > = 1.0;
uniform float Vehicle_Exhaust_Smoke < ui_label = "Exhaust Smoke Density"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform float Vehicle_Exhaust_Smoke_Scale < ui_label = "Exhaust Smoke Scale"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform int Vehicle_Burnout < ui_type = "slider"; ui_label = "Vehicle Burnout"; ui_min = 0; ui_max = 2; ui_category = "PARTICLE FX"; ui_spacing = 2; > = 1;
uniform int Vehicle_Burnout_Density < ui_type = "slider"; ui_label = "Vehicle Burnout Density"; ui_min = 0; ui_max = 2; ui_category = "PARTICLE FX"; > = 0;
uniform float Vehicle_Burnout_Bright < ui_label = "Vehicle Burnout Bright"; ui_type = "drag"; ui_min = 0.001; ui_max = 3.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform float Vehicle_Burnout_Scale < ui_label = "Vehicle Burnout Scale"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform float3 Vehicle_Burnout_Color < ui_label = "Vehicle Burnout Color"; ui_type = "color"; ui_category = "PARTICLE FX"; > = float3(1.0, 1.0, 1.0);
uniform float Vehicle_Engine_Fire < ui_label = "Damaged Engine Fire Intensity"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PARTICLE FX"; ui_spacing = 2; > = 1.0;
uniform float Vehicle_Engine_Smoke < ui_label = "Damaged Engine Smoke Density"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform float Vehicle_Engine_Smoke_Scale < ui_label = "Damaged Engine Smoke Scale"; ui_type = "drag"; ui_min = 0.001; ui_max = 8.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform float Gun_Muzzle_Flash_Intensity < ui_label = "Gun Muzzle Flash Intensity"; ui_type = "drag"; ui_min = 0.001; ui_max = 6.0; ui_category = "PARTICLE FX"; ui_spacing = 2; > = 1.0;
uniform float Gun_Muzzle_Flash_Scale < ui_label = "Gun Muzzle Flash Scale"; ui_type = "drag"; ui_min = 0.001; ui_max = 8.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform float Gun_Muzzle_Smoke_Bright < ui_label = "Gun Muzzle Smoke Bright"; ui_type = "drag"; ui_min = 0.001; ui_max = 6.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform float Gun_Muzzle_Smoke_Density < ui_label = "Gun Muzzle Smoke Density"; ui_type = "drag"; ui_min = 0.001; ui_max = 6.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform float Gun_Muzzle_Smoke_Scale < ui_label = "Gun Muzzle Smoke Scale"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform float Explosion_Intensity < ui_label = "Explosion Intensity"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PARTICLE FX"; ui_spacing = 2; > = 1.0;
uniform float Explosion_Scale < ui_label = "Explosion Scale"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PARTICLE FX"; > = 1.0;
uniform int Blood_Puddle < ui_type = "combo"; ui_label = "Blood Puddle"; ui_items = "Texture Glossiness\0Procedural\0Texture Glossiness + Procedural\0"; ui_category = "PARTICLE FX"; ui_spacing = 2; > = 0;
uniform int sep5 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform int Vegetation < ui_label = "Vegetation Season"; ui_category = "VEGETATION"; ui_type = "combo"; ui_items = "Spring\0Summer (default-vanilla)\0Autumn1\0Autumn2\0Autumn3\0Autumn4\0Winter\0"; > = 1;
uniform bool Vegetation_Complex_Shadows < ui_label = "Vegetation Complex Shadows"; ui_category = "VEGETATION"; ui_tooltip = "Experimental Complex Shadows on grass / bushes / trees"; > = false;
uniform int sep6 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool Volumetric_Clouds < ui_label = "Volumetric Clouds"; > = false;
#if __VENDOR__ == 0x1002
uniform bool Volumetric_Clouds_AMD_Fix < ui_label = "Volumetric Clouds - AMD Fix"; ui_tooltip = "do not enable unless you experience issues with clouds\nRequires disable and enable clouds again"; > = false;
#endif
uniform bool Dusty_Wind < ui_label = "Volumetric Dusty Wind - Desert Area - Blizzard - Snow"; ui_tooltip = "Experimental effect with sun&moon scattering light"; > = false;
uniform int sep7 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool Enable_Motion_Blur_while_driving_in_FPV < ui_label = "Enable MotionBlur while driving in FPV"; ui_category = "MOTION BLUR"; > = false;
uniform bool Disable_Motion_Blur < ui_label = "Disable MotionBlur"; ui_category = "MOTION BLUR"; > = false;
#if __APPLICATION__ != -1664759584
uniform int sep8 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool Server_Overbloom_Fix < ui_label = "Servers with Over-Bloom Fix"; ui_tooltip = "Workaround for FiveM servers that overwrites visualsettings.dat causing too much bloom on bright surfaces"; ui_category = "FiveM workarounds"; > = false;
uniform bool FiveM_Outline_Crash < ui_label = "Workaround for FiveM Outline Crash"; ui_tooltip = "Workaround to avoid crashes in FiveM servers using: 'SetEntityDrawOutline' known for crashing the game\nPlease disable it once the Outline function is not used anymore because many features won't work with it"; ui_category = "FiveM workarounds"; > = false;
#endif
uniform int sep9 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool GPLens < ui_label = "GPLens"; ui_category = "LENS FLARE"; ui_tooltip = "GP's lensflare"; > = false;
uniform bool SixLens < ui_label = "SixLens"; ui_category = "LENS FLARE"; ui_tooltip = "High Quality Lensflare partially inspired by 'Six' #1 trailer"; ui_spacing = 3; > = false;
uniform float SixLensIntensity < ui_label = "Lens Flare Intensity"; ui_category = "LENS FLARE"; ui_type = "drag"; ui_min = 0.0; ui_max = 8.0; > = 1.0;
uniform float lightStreak < ui_label = "LightStreak Intensity"; ui_category = "LENS FLARE"; ui_type = "drag"; ui_min = 0.0; ui_max = 7.0; ui_tooltip = "Light Streak effect produced by bright surfaces radiating outwards in a star-like pattern"; > = 1.0;
uniform float lightStreakThreshold < ui_label = "LightStreak Threshold"; ui_category = "LENS FLARE"; ui_type = "drag"; ui_min = 0.01; ui_max = 6.0; > = 1.0;
uniform int lightStreakNumber < ui_label = "LightStreaks Number"; ui_category = "LENS FLARE"; ui_type = "slider"; ui_tooltip = "Number of rays/streaks produced"; ui_min = 3; ui_max = 12; > = 10;
uniform float lightStreakLength < ui_label = "LightStreak Length"; ui_category = "LENS FLARE"; ui_type = "drag"; ui_min = 0.01; ui_max = 8.0; ui_tooltip = "Light Streak Size/Length"; > = 1.0;
uniform float lightStreakTaps < ui_label = "LightStreak Taps"; ui_category = "LENS FLARE"; ui_type = "drag"; ui_min = 0.01; ui_max = 1.0; ui_tooltip = "Increasing it may be heavy on performance"; > = 0.25;
uniform float Anamorphic_Lens_Global < ui_label = "Anamorphic flare Global"; ui_category = "LENS FLARE"; ui_type = "drag"; ui_min = 0.0; ui_max = 60.0; ui_tooltip = "Anamorphic lensflare produced by bright spots"; ui_spacing = 3; > = 1.0;
uniform float Anamorphic_Lens_Lights < ui_label = "Anamorphic flare Lights"; ui_category = "LENS FLARE"; ui_type = "drag"; ui_min = 0.0; ui_max = 600.0; ui_tooltip = "Anamorphic lensflare produced by lights sources"; > = 1.0;
uniform float Anamorphic_Lens_EmergencyLights < ui_label = "Anamorphic flare Emergency Lights"; ui_category = "LENS FLARE"; ui_type = "drag"; ui_min = 0.0; ui_max = 600.0; ui_tooltip = "Anamorphic lensflare produced by emergency lights"; > = 1.9;
uniform int Anamorphic_Style < ui_label = "Anamorphic Style"; ui_category = "LENS FLARE"; ui_type = "combo"; ui_items = "default\0QuantV\0"; ui_tooltip = "Anamorphic lensflare (lights) Style"; > = 1;
uniform int sep10 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool Brighter_Nights < ui_label = "Brighter Nights"; > = false;
uniform bool Smooth_Night_Shadows < ui_label = "Smooth Night Shadows"; > = false;
uniform int sep11 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool Starfield_Galaxy < ui_label = "Starfield Galaxy"; ui_category = "STARFIELD"; > = false;
uniform bool Galaxy_RGB < ui_label = "Dynamic Color RGB"; ui_category = "STARFIELD"; > = false;
uniform bool Galaxy_Enable_Tint < ui_label = "Manual Color Tint"; ui_category = "STARFIELD"; > = false;
uniform float4 Galaxy_Tint < ui_label = "Color Tint"; ui_type = "color"; ui_category = "STARFIELD"; > = float4(0.4, 0.2, 0.02, 0.6);
uniform float Galaxy_Rotation < ui_label = "Rotation Position"; ui_type = "slider"; ui_min = -3.1416; ui_max = 3.1416; ui_category = "STARFIELD"; > = 0.0;
uniform int sep12 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform float HDR_Reflection < ui_label = "HDR Reflections"; ui_type = "drag"; ui_min = 0.0; ui_max = 10.0; ui_category = "ROADS & SURFACES";  ui_tooltip = "Environment HDR Reflections"; > = 1.0;
uniform bool Road < ui_label = "Roads Overhaul"; ui_category = "ROADS & SURFACES"; ui_tooltip = "Work In Progress..."; > = false;
uniform int Wet_Roads < ui_label = "Wet Roads"; ui_category = "ROADS & SURFACES"; ui_type = "combo"; ui_items = "very low wetness\0low wetness\0default wetness (vanilla)\0high wetness\0very high wetness\0"; > = 2;
uniform bool Rain_Reflection < ui_label = "Rain Puddle Reflections (SSR)"; ui_category = "ROADS & SURFACES"; ui_tooltip = "Screen Space Reflections (W.I.P)"; > = true;
uniform float Rain_Reflection_Quality < ui_label = "Reflection Quality"; ui_type = "drag"; ui_min = 0.0; ui_max = 5.0; ui_category = "ROADS & SURFACES"; > = 1.0;
uniform float Puddle_Scale < ui_label = "Puddle Scale"; ui_type = "drag"; ui_min = 0.0; ui_max = 10.0; ui_category = "ROADS & SURFACES"; > = 1.0;
uniform bool Puddles_Covering_Full_Ground < ui_label = "Puddles Covering Full Ground"; ui_category = "ROADS & SURFACES"; ui_tooltip = "Floor looks like a mirror when its raining"; > = false;
uniform int Snow < ui_label = "Snow Vanilla"; ui_type = "combo"; ui_items = "Vanilla\0Snow Mod - Full Cover\0Snow Mod - Melting Puddles on Roads\0Snow Mod - Clean Roads\0"; ui_category = "ROADS & SURFACES"; ui_tooltip = "Modification of existing snow\nRequires game snow enabled"; > = 0;
uniform int QV_Snow < ui_label = "Snow QV (additional)"; ui_type = "combo"; ui_items = "Disabled\0Full Cover\0Clean Roads\0Melting Puddles on Roads\0"; ui_category = "ROADS & SURFACES"; ui_tooltip = "Custom Snow Shader (Independent)\nDoes not requires game snow"; > = 0;
uniform float2 Snow_Altitude_FadeInOut < ui_label = "Altitude Fade In & Out"; ui_type = "drag"; ui_min = 0.0; ui_category = "ROADS & SURFACES"; ui_tooltip = "Fade IN    |    Fade OUT\n\nexample:\n1.2-1.3 beach shore\n360-560 mountain top"; > = float2(1.2, 1.3);
uniform int sep13 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform float SSS_Intensity < ui_label = "SSS Intensity"; ui_type = "drag"; ui_min = 0.001; ui_max = 1.0; ui_category = "PLAYER PED"; ui_tooltip = "Subsurface Scattering Intensity"; > = 0.9;
uniform float SSS_Radius < ui_label = "SSS Radius"; ui_type = "drag"; ui_min = 0.001; ui_max = 8.0; ui_category = "PLAYER PED"; ui_tooltip = "Subsurface Scattering Radius"; > = 1.0;
uniform float SSS_Reflection < ui_label = "SSS Reflection"; ui_type = "drag"; ui_min = 0.001; ui_max = 20.0; ui_category = "PLAYER PED"; ui_tooltip = "Subsurface Scattering Reflection"; > = 1.0;
uniform float pedBumpiness < ui_label = "Ped Skin Bumpiness"; ui_type = "drag"; ui_min = 0.001; ui_max = 3.0; ui_category = "PLAYER PED"; ui_tooltip = "Skin Bumpiness"; > = 1.0;
uniform float skinDetail < ui_label = "Ped Skin Detail"; ui_type = "drag"; ui_min = 0.001; ui_max = 4.0; ui_category = "PLAYER PED"; ui_tooltip = "Skin / Facial Detail"; > = 1.0;
uniform float skinSpecIntensity < ui_label = "Ped Skin Specular Intensity"; ui_type = "drag"; ui_min = 0.001; ui_max = 1000.0; ui_category = "PLAYER PED"; ui_tooltip = "Skin Specular Intensity"; > = 1.0;
uniform float skinSpecFresnel < ui_label = "Ped Skin Specular Fresnel"; ui_type = "drag"; ui_min = 0.001; ui_max = 10.0; ui_category = "PLAYER PED"; ui_tooltip = "Skin Specular Fresnel"; > = 1.0;
uniform int sep14 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool Disable_Chromatic_Aberration_n_Distortion < ui_label = "Disable ChromaticAberration & Distortion"; ui_category = "DISABLE FEATURES"; > = false;
uniform bool Disable_Volume_Lights < ui_label = "Disable Volume Lights"; ui_category = "DISABLE FEATURES"; > = false;
uniform bool Disable_Weapon_Reticle < ui_label = "Disable Weapon Reticle"; ui_category = "DISABLE FEATURES"; > = false;
uniform bool Blood_Remove < ui_label = "Disable Screen Blood effect"; ui_category = "DISABLE FEATURES"; > = false;
uniform int sep15 < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;
uniform bool Skip_HUD < ui_label = "Skip HUD Elements (W.I.P)"; ui_tooltip = "wait 10 seconds after enabling it"; > = false;
uniform int sepn < ui_label = " "; ui_type = "radio"; ui_text = "\n\n"; noedit = true; >;

uniform float timer < source = "timer"; >;
uniform bool overlay_open < source = "overlay_open"; >;
uniform float wasdf < hidden = true; >;
uniform int waasdf < hidden = true; > = __APPLICATION__;
static const float2 rcpFrame = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
uniform float4 QuantV_1 < hidden = true; >;
uniform float4 QuantV_2 < hidden = true; >;
uniform float4 QuantV_3 < hidden = true; >;

texture BackBufferTex : COLOR;
texture DepthBufferTex : DEPTH;
sampler BackBuffer
{
    Texture = BackBufferTex;
};
sampler DepthBuffer
{
    Texture = DepthBufferTex;
};
texture2D TextureOriginal
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
};
sampler2D sTextureOriginal
{
    Texture = TextureOriginal;
};
texture2D t8
{
    Width = BUFFER_WIDTH * 0.125;
    Height = BUFFER_HEIGHT * 0.125;
    Format = RGB10A2;
};
sampler2D st8
{
    Texture = t8;
};
texture shape < source = "QuantV/LensRain.png"; >
{
    Width = 1024;
    Height = 1024;
    Format = R8;
};
sampler2D shapes
{
    Texture = shape;
};
sampler BackBufferM
{
    Texture = BackBufferTex;
    AddressU = MIRROR;
    AddressV = MIRROR;
};
texture dd < source = "QuantV/18673.png"; >
{
    Width = 512;
    Height = 512;
};
sampler2D sdd
{
    Texture = dd;
    AddressU = WRAP;
    AddressV = WRAP;
};
#if (__RENDERER__ == 0xa000 || __RENDERER__ == 0xa100)
texture dx < source = "QuantV/dx.png"; >
{
    Width = 444;
    Height = 45;
};
sampler2D dxbuffer
{
    Texture = dx;
};
#endif

#define sngbool QuantV_1.x > 0.001 && Screen_Raindrops_NFS
#define t (timer > QuantV_1.y * 10000000.0 - 100.0) * saturate(tex2Dfetch(sTextureOriginal, 0).y) * (timer < QuantV_1.y * 10000000.0 + 100.0) * 23.9999
#define df(x,y) lerp(lerp(y, x, smoothstep(5.0, 5.5, t)), y, smoothstep(21.0, 21.5, t))
#define srs 0.05 * Screen_Raindrops_Scale
static const uint2 seed[72] = { uint2(150, 183), uint2(140, 181), uint2(148, 157), uint2(218, 190), uint2(184, 170), uint2(212, 205), uint2(137, 188), uint2(125, 138), uint2(129, 156), uint2(164, 198), uint2(127, 130), uint2(152, 175), uint2(163, 111), uint2(147, 191), uint2(117, 161), uint2(124, 197), uint2(220, 216), uint2(106, 193), uint2(176, 143), uint2(173, 171), uint2(209, 217), uint2(103, 204), uint2(141, 177), uint2(159, 122), uint2(201, 207), uint2(134, 200), uint2(211, 146), uint2(195, 105), uint2(196, 165), uint2(107, 100), uint2(178, 219), uint2(186, 135), uint2(101, 208), uint2(123, 144), uint2(162, 202), uint2(115, 214), uint2(149, 153), uint2(108, 155), uint2(168, 187), uint2(136, 133), uint2(145, 199), uint2(110, 126), uint2(121, 180), uint2(215, 119), uint2(172, 128), uint2(160, 112), uint2(203, 194), uint2(139, 192), uint2(113, 182), uint2(169, 174), uint2(154, 131), uint2(206, 104), uint2(142, 151), uint2(116, 132), uint2(166, 120), uint2(102, 109), uint2(179, 167), uint2(118, 158), uint2(210, 213), uint2(114, 185), uint2(241, 245), uint2(235, 231), uint2(221, 232), uint2(222, 238), uint2(239, 226), uint2(230, 234), uint2(240, 242), uint2(229, 237), uint2(223, 225), uint2(233, 243), uint2(228, 244), uint2(224, 236) };
void VS_Q_00(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
    texcoord.x = (id == 2) ? 2.0 : 0.0;
    texcoord.y = (id == 1) ? 2.0 : 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}
void VS_Q_01(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD, out float4 t1[3] : TEXCOORD1)
{
    texcoord.x = (id == 2) ? 2.0 : 0.0;
    texcoord.y = (id == 1) ? 2.0 : 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    float sz = 4.0 * 4.0 * 1.0;
    texcoord *= 1.0 - saturate(tex2Dfetch(sTextureOriginal, int2(0, 1)).y);
    t1[0].xy = texcoord + float2(rcpFrame.x * sz, 0.0);
    t1[0].zw = texcoord + rcpFrame * sz * float2(0.5, 0.8660254);
    t1[1].xy = texcoord + rcpFrame * sz * float2(-0.5, 0.8660254);
    t1[1].zw = texcoord + float2(-rcpFrame.x * sz, 0.0);
    t1[2].xy = texcoord + rcpFrame * sz * float2(-0.5, -0.8660254);
    t1[2].zw = texcoord + rcpFrame * sz * float2(0.5, -0.8660254);
}
void VS_Q_02(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD, out float4 t1[3] : TEXCOORD1, out nointerpolation float2 t2 : TEXCOORD4, out nointerpolation float4 t3 : TEXCOORD5)
{
    texcoord.x = (id == 2) ? 2.0 : 0.0;
    texcoord.y = (id == 1) ? 2.0 : 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    float sz = 4.0 * 4.0 * 3.0;
    t1[0].xy = texcoord + float2(rcpFrame.x * sz, 0.0);
    t1[0].zw = texcoord + rcpFrame * sz * float2(0.5, 0.8660254);
    t1[1].xy = texcoord + rcpFrame * sz * float2(-0.5, 0.8660254);
    t1[1].zw = texcoord + float2(-rcpFrame.x * sz, 0.0);
    t1[2].xy = texcoord + rcpFrame * sz * float2(-0.5, -0.8660254);
    t1[2].zw = texcoord + rcpFrame * sz * float2(0.5, -0.8660254);
    t2.xy = tex2Dfetch(sTextureOriginal, 0).yz;
    t3.x = df(3.98, 0.55);
    t3.y = (QuantV_1.w || Screen_Frozen_Snow_ForceEnable);
    t3.z = t3.y * Screen_Frozen_Snow * (1.0 - saturate(tex2Dfetch(sTextureOriginal, int2(0, 1)).y));
    t3.w = 0.0;
}
void func_qr(float2 texcoord, inout float4 t1[30], uint index)
{
    float2 sz[30];
    float2 fo[30];
    float ist = ((timer > QuantV_1.y * 10000000.0 - 100.0) && (timer < QuantV_1.y * 10000000.0 + 100.0)) * lerp(0.741, 1.049, saturate(tex2Dfetch(sTextureOriginal, 0).x));
    [unroll] for (int i = 0; i < 30; ++i)
    {
        float sd = QuantV_1.y * seed[i + index].x * 21.0;
        float2 noise = frac(sin(floor(sd) * 12.9898 + seed[i + index] * 78.233) * 43758.5453);
        float2 coord = noise - float2(0.499, 0.52);
        float vignt = length(coord);
        sz[i] = (1.0 + frac(sd) * 0.7 + 0.3) * 0.05;
        sz[i].x *= (rcpFrame.x / rcpFrame.y);
        sz[i] *= lerp(1.0, 0.4, noise.y) * (1.0 - saturate(tex2Dfetch(sTextureOriginal, int2(0, 1)).y));
        fo[i] = coord / vignt * lerp(0.225, float2(1.0, 0.5), vignt * 2.0);
        fo[i].x *= (rcpFrame.x / rcpFrame.y);
        float2 Vec; sincos(seed[i + index].x, Vec.y, Vec.x);
        t1[i].xy = mul(float2x2(Vec.x, -Vec.y, Vec.y, Vec.x), ((texcoord * ist + (1.0 - ist) * float2(0.5, 0.19)) - float2(0.499, 0.52) - fo[i]) / sz[i]);
        t1[i].zw = (float2(4.0 * frac(i * 0.25), floor(i * 0.25)) + t1[i].xy + 0.5) * 0.25;
    }
}
void VS_Q_0(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD, out float4 t1[30] : TEXCOORD1)
{
    texcoord.x = (id == 2) ? 2.0 : 0.0;
    texcoord.y = (id == 1) ? 2.0 : 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    [unroll] for (int i = 0; i < 30; ++i) t1[i] = 0.0;
    if (sngbool) func_qr(texcoord, t1, 0);
}
void VS_Q_1(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD, out float4 t1[30] : TEXCOORD1)
{
    texcoord.x = (id == 2) ? 2.0 : 0.0;
    texcoord.y = (id == 1) ? 2.0 : 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    [unroll] for (int i = 0; i < 30; ++i) t1[i] = 0.0;
    if (sngbool) func_qr(texcoord, t1, 30);
}
float4 PS_Q_00(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float d = tex2Dlod(DepthBuffer, float4(texcoord, 0.0, 0.0)).x == 0.0;
    d += tex2Dlod(DepthBuffer, float4(texcoord, 0.0, 0.0), int2(1, 0)).x == 0.0;
    d += tex2Dlod(DepthBuffer, float4(texcoord, 0.0, 0.0), int2(0, 1)).x == 0.0;
    d += tex2Dlod(DepthBuffer, float4(texcoord, 0.0, 0.0), int2(-1, 0)).x == 0.0;
    d += tex2Dlod(DepthBuffer, float4(texcoord, 0.0, 0.0), int2(0, -1)).x == 0.0;
    d = d * 0.2;
    return float4(tex2Dlod(BackBuffer, float4(texcoord, 0.0, 0.0)).xyz, d);
}
float4 PS_Q_01(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return float4(tex2Dlod(BackBuffer, float4(texcoord, 0.0, 0.0)).xyz, 0.0);
}
float4 PS_Q_1(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return tex2Dlod(BackBuffer, float4(texcoord, 0.0, 0.0));
}
float4 PS_Q_2(float4 position : SV_Position, float2 texcoord : TEXCOORD, float4 t1[30] : TEXCOORD1) : SV_Target
{
    float4 res = tex2Dlod(BackBuffer, float4(texcoord, 0.0, 0.0));
    if (sngbool)
    {
        [unroll] for (int i = 0; i < 30; ++i)
        {
            float lens = tex2Dlod(shapes, float4(t1[i].zw, 0.0, 0.0)).x;
            bool2 k1 = (0.5 >= abs(t1[i].xy));
            res.w += lens * k1.x * k1.y * (1.0 - frac(QuantV_1.y * seed[i].x * 21.0)) * (1.0 - saturate(tex2Dfetch(sTextureOriginal, int2(0, 1)).z)) * lerp(lerp(1.0, 0.2, smoothstep(4.9, 5.4, t)), 1.0, smoothstep(20.5, 21.1, t));
        }
        res.w *= QuantV_1.x;
    }
    return res;
}
float4 PS_Q_3(float4 position : SV_Position, float2 texcoord : TEXCOORD, float4 t1[30] : TEXCOORD1) : SV_Target
{
    float4 res = tex2Dlod(BackBuffer, float4(texcoord, 0.0, 0.0));
    if (sngbool)
    {
        [unroll] for (int i = 0; i < 30; ++i)
        {
            float lens = tex2Dlod(shapes, float4(t1[i].zw, 0.0, 0.0)).x;
            bool2 k1 = (0.5 >= abs(t1[i].xy));
            res.w += lens * k1.x * k1.y * (1.0 - frac(QuantV_1.y * seed[i + 30].x * 21.0)) * (1.0 - saturate(tex2Dfetch(sTextureOriginal, int2(0, 1)).z)) * lerp(lerp(1.0, 0.2, smoothstep(4.9, 5.4, t)), 1.0, smoothstep(20.5, 21.1, t));
        }
        res.w *= QuantV_1.x;
    }
    return res;
}
float4 PS_Q_bb(float4 position : SV_Position, float2 texcoord : TEXCOORD, float4 t1[3] : TEXCOORD1) : SV_Target
{
    float3 res = tex2Dlod(BackBuffer, float4(texcoord, 0, 0)).xyz;
    if (sngbool)
    {
        float3 b = res;
        b += tex2Dlod(BackBuffer, float4(t1[0].xy, 0, 0)).xyz;
        b += tex2Dlod(BackBuffer, float4(t1[0].zw, 0, 0)).xyz;
        b += tex2Dlod(BackBuffer, float4(t1[1].xy, 0, 0)).xyz;
        b += tex2Dlod(BackBuffer, float4(t1[1].zw, 0, 0)).xyz;
        b += tex2Dlod(BackBuffer, float4(t1[2].xy, 0, 0)).xyz;
        b += tex2Dlod(BackBuffer, float4(t1[2].zw, 0, 0)).xyz;
        float3 fl = tex2Dlod(BackBufferM, float4(texcoord + (0.5 - texcoord) * 1.5, 0, 0)).xyz;
        fl = smoothstep(fl, 0.0, saturate(0.97 - dot(fl, float3(0.2126, 0.7152, 0.0722))));
        b += lerp(dot(fl, float3(0.2126, 0.7152, 0.0722)), fl, 1.5) * 2.5;
        res = b * (1.0 / 7.0);
    }
    return float4(res, 0.0);
}
float4 PS_Q_mx2(float4 position : SV_Position, float2 texcoord : TEXCOORD, float4 t1[3] : TEXCOORD1, nointerpolation float2 t2 : TEXCOORD4, nointerpolation float4 t3 : TEXCOORD5) : SV_Target
{
    float4 res = tex2Dlod(BackBuffer, float4(texcoord, 0, 0));
    res.w = ((position.x < 1) && (position.y < 1)) ? t2.y : t2.x;
    if ((position.x < 1) && (position.y < 3)) res.xyz = tex2Dlod(BackBuffer, float4(texcoord, 0.0, 0.0), int2(1, 0)).xyz;
    if (sngbool)
    {
        float3 b = tex2Dlod(st8, float4(texcoord, 0, 0)).xyz;
        b += tex2Dlod(st8, float4(t1[0].xy, 0, 0)).xyz;
        b += tex2Dlod(st8, float4(t1[0].zw, 0, 0)).xyz;
        b += tex2Dlod(st8, float4(t1[1].xy, 0, 0)).xyz;
        b += tex2Dlod(st8, float4(t1[1].zw, 0, 0)).xyz;
        b += tex2Dlod(st8, float4(t1[2].xy, 0, 0)).xyz;
        b += tex2Dlod(st8, float4(t1[2].zw, 0, 0)).xyz;
        b *= (1.0 / 7.0);
        b += tex2Dlod(st8, float4(texcoord + float2(-0.06, 0.35), 0, 0)).xyz * 0.3;
        b += tex2Dlod(st8, float4(texcoord + float2(0.01, -0.25), 0, 0)).xyz * 0.2;
        b = pow(saturate(b), 1.2);
        float dt = tex2Dlod(BackBuffer, float4(texcoord, 0, 0)).w;
        float rr = pow(tex2Dlod(BackBuffer, float4(texcoord, 0, 0), int2(1, 0)).w, 2.0);
        float uu = pow(tex2Dlod(BackBuffer, float4(texcoord, 0, 0), int2(0, 1)).w, 2.0);
        b = saturate(lerp(dot(b, float3(0.2126, 0.7152, 0.0722)), b, 2.5) * dt * 2.9);
        res.xyz = lerp(tex2Dlod(sTextureOriginal, float4(texcoord + saturate(dt - float2(rr, uu)) * 0.02, 0, 0)).xyz, 0.96, b);
    }
    if (Screen_Frozen_Snow)
    {
        float3 tex = tex2Dlod(sdd, float4(texcoord * 2.0 - timer * 0.0000002, 0, 0)).xyz;
        tex.xy -= 0.5;
        float vig = pow(abs(1.0 - smoothstep(0.975, 0.05, length(texcoord - 0.5) * t3.y)), 7.0) * t3.z;
        float noise = frac(sin(dot(texcoord, float2(12.9898, 78.233))) * 43758.5453);
        noise *= 0.025 * vig + tex.z * vig * t3.x;
        res.xyz = tex2Dlod(BackBufferM, float4(texcoord + tex.xy * vig, 0.0, 0.0)).xyz;
        res.xyz = lerp(res.xyz, 1.0, saturate(noise * 5.0));
    }
    return res;
}
#if (__RENDERER__ == 0xa000 || __RENDERER__ == 0xa100)
float4 PS_dx(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 res = tex2Dlod(BackBuffer, float4(texcoord, 0.0, 0.0));
    float2 scale = float2(BUFFER_WIDTH, BUFFER_HEIGHT) * float2(1.0 / 444.0, 1.0 / 45.0);
    float2 uv = (texcoord - 0.5) + 0.5 / scale;
    float3 m = tex2Dlod(dxbuffer, float4(uv * scale, 0.0, 0.0)).xyz;
    if (timer < 20000.0) res.xyz = lerp(res.xyz, m, saturate(m * rcp(timer) * 33333.0));
    return res;
}
#endif

technique QuantV < ui_tooltip = "QuantV"; >
{
#if (__RENDERER__ == 0xa000 || __RENDERER__ == 0xa100)
    pass p0
    {
        VertexShader = VS_Q_00;
        PixelShader = PS_dx;
    }
#else
    pass p0
    {
        VertexShader = VS_Q_00;
        PixelShader = PS_Q_00;
        RenderTarget = TextureOriginal;
    }
    pass p1
    {
        VertexShader = VS_Q_00;
        PixelShader = PS_Q_01;
    }
    pass p2
    {
        VertexShader = VS_Q_0;
        PixelShader = PS_Q_2;
    }
    pass p3
    {
        VertexShader = VS_Q_1;
        PixelShader = PS_Q_3;
    }
    pass p4
    {
        VertexShader = VS_Q_01;
        PixelShader = PS_Q_bb;
        RenderTarget = t8;
    }
    pass p5
    {
        VertexShader = VS_Q_02;
        PixelShader = PS_Q_mx2;
    }
#endif
}
