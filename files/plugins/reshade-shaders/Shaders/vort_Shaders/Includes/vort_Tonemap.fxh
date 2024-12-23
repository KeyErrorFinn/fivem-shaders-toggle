/*******************************************************************************
    Author: Vortigern

    License: MIT, Copyright (c) 2023 Vortigern

    MIT License

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************

#pragma once
#include "Includes/vort_Defs.fxh"

/*******************************************************************************
    Functions
*******************************************************************************/

float3 ApplyLottes(float3 c)
{
    c = (1.02 * c) * RCP(1.0 + Max3(c.r, c.g, c.b));
    c = saturate(c);

    return c;
}

float3 InverseLottes(float3 c)
{
    c = saturate(c);
    c = c * RCP(1.02 - Max3(c.r, c.g, c.b));

    return c;
}

// https://www.desmos.com/calculator/mpslmho5wp
float3 ApplyReinhard(float3 c)
{
    c = (1.02 * c) * RCP(1.0 + RGBToYCbCrLumi(c));
    c = saturate(c);

    return c;
}

// https://www.desmos.com/calculator/mpslmho5wp
float3 InverseReinhard(float3 c)
{
    c = saturate(c);
    c = c * RCP(1.02 - RGBToYCbCrLumi(c));

    return c;
}

// https://www.desmos.com/calculator/mpslmho5wp
float3 ApplyStanard(float3 c)
{
    static const float k = sqrt(4.0 / 27.0);

    c = min(50.0, c);
    c = c * sqrt(c);
    c = c * RCP(k + RGBToYCbCrLumi(c));
    c = saturate(c);

    return c;
}

// https://www.desmos.com/calculator/mpslmho5wp
float3 InverseStanard(float3 c)
{
    static const float k = sqrt(4.0 / 27.0);
    static const float tt = 2.0 / 3.0;

    c = saturate(c);
    c = (k * c) * RCP(1.0 - RGBToYCbCrLumi(c));
    c = POW(c, tt);
    c = min(50.0, c);

    return c;
}
