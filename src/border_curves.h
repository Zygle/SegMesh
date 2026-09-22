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
    double fitTolerance = 0.01;
    double drawingTolerance = 0.001;
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
    uint32_t cubicPieceCount = 0;
    uint32_t renderedPointCount = 0;
    double maxFitError = 0.0;
    double rmsFitError = 0.0;
    bool drawingSubdivisionCapped = false;
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
    uint32_t cubicPieceCount = 0;
    uint32_t drawingCappedChainCount = 0;
    double maxFitError = 0.0;
    double rmsFitError = 0.0;
};

Float3 evaluateBezier(std::span<const Float3> controlPoints, double t);

bool fitEndpointConstrainedCubic(
    std::span<const Float3> points,
    std::array<Float3, 4>& outControlPoints,
    BorderCurveFitStats& outStats,
    std::string& error
);

bool buildSegmentBorderOverlay(
    const CpuMesh& mesh,
    std::span<const uint32_t> triangleLabels,
    const BorderCurveSettings& settings,
    BorderCurveOverlay& outOverlay,
    std::string& error
);
}
