#pragma once

#include "mesh_types.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace segmesh
{
struct BorderCurveSettings
{
    double tolerance = 0.002;
    uint32_t maxDegree = 64;
};

struct BorderCurveFitStats
{
    double maxError = 0.0;
    double rmsError = 0.0;
};

struct BorderCurveChainStats
{
    uint32_t firstLabel = 0;
    uint32_t secondLabel = 0;
    uint32_t sourcePointCount = 0;
    uint32_t fittedPointCount = 0;
    uint32_t finalDegree = 0;
    double maxFitError = 0.0;
    double rmsFitError = 0.0;
    bool degreeCapped = false;
    bool closedLoopArc = false;
};

struct BorderCurveOverlay
{
    std::vector<Float3> vertices;
    std::vector<uint32_t> indices;
    std::vector<Float3> controlVertices;
    std::vector<uint32_t> controlIndices;
    std::vector<BorderCurveChainStats> chains;

    uint32_t boundaryEdgeCount = 0;
    uint32_t skippedChainCount = 0;
    uint32_t cappedChainCount = 0;
    double maxFitError = 0.0;
    double rmsFitError = 0.0;
};

Float3 evaluateBezier(std::span<const Float3> controlPoints, double t);

std::vector<Float3> elevateBezierDegree(std::span<const Float3> controlPoints);

bool fitEndpointConstrainedCubic(
    std::span<const Float3> points,
    std::array<Float3, 4>& outControlPoints,
    BorderCurveFitStats& outStats,
    std::string& error
);

std::vector<Float3> adaptiveDegreeElevate(
    const std::array<Float3, 4>& cubicControlPoints,
    const BorderCurveSettings& settings,
    bool& outDegreeCapped
);

bool buildSegmentBorderOverlay(
    const CpuMesh& mesh,
    std::span<const uint32_t> triangleLabels,
    const BorderCurveSettings& settings,
    BorderCurveOverlay& outOverlay,
    std::string& error
);
}
