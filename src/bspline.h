#pragma once

#include "mesh_types.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace segmesh
{
struct CubicBSpline
{
    static constexpr uint32_t degree = 3;
    std::vector<Float3> controlPoints;
    std::vector<double> knots;
    bool periodic = false;
};

struct BSplineFitSettings
{
    double fitTolerance = 0.01;
    double minimumSpanLength = 0.0;
    uint32_t maxControlPointCount = 128;
};

struct BSplineFitStats
{
    double maxError = 0.0;
    double maxSampleError = 0.0;
    double rmsError = 0.0;
    uint32_t refinementCount = 0;
    bool refinementCapped = false;
};

struct BSplineBezierSpan
{
    std::array<Float3, 4> controlPoints{};
    double firstParameter = 0.0;
    double lastParameter = 1.0;
};

bool evaluateBSpline(
    const CubicBSpline& curve,
    double parameter,
    Float3& outPoint,
    std::string& error
);

bool convertBSplineToBezierSpans(
    const CubicBSpline& curve,
    std::vector<BSplineBezierSpan>& outSpans,
    std::string& error
);

bool fitAdaptiveCubicBSpline(
    std::span<const Float3> guidePoints,
    bool periodic,
    const BSplineFitSettings& settings,
    CubicBSpline& outCurve,
    BSplineFitStats& outStats,
    std::string& error
);
}
