#pragma once

#include "bezier.h"
#include "border_trace.h"
#include "mesh_types.h"

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

// Owned by a session. Advance meshGeneration whenever mesh geometry/topology changes.
class BorderCurveCache
{
public:
    void invalidate();
    bool buildOverlay(
        const CpuMesh& mesh,
        uint64_t meshGeneration,
        std::span<const uint32_t> triangleLabels,
        const BorderCurveSettings& settings,
        BorderCurveOverlay& outOverlay,
        std::string& error
    );

private:
    struct FittedChain
    {
        std::vector<detail::CubicPiece> pieces;
        BorderCurveChainStats stats;
    };

    bool fitTracedBorders(const CpuMesh& mesh, double fitTolerance, std::string& error);
    bool sampleFittedBorders(
        double drawingTolerance,
        BorderCurveOverlay& outOverlay,
        std::string& error
    ) const;

    bool traceValid_ = false;
    bool fitValid_ = false;
    uint64_t meshGeneration_ = 0;
    std::vector<uint32_t> triangleLabels_;
    std::vector<detail::TracedChain> tracedChains_;
    uint32_t boundaryEdgeCount_ = 0;
    double fitTolerance_ = 0.0;
    std::vector<FittedChain> fittedChains_;
    uint32_t skippedChainCount_ = 0;
    uint32_t cubicPieceCount_ = 0;
    double maxFitError_ = 0.0;
    double rmsFitError_ = 0.0;
};

bool buildSegmentBorderOverlay(
    const CpuMesh& mesh,
    std::span<const uint32_t> triangleLabels,
    const BorderCurveSettings& settings,
    BorderCurveOverlay& outOverlay,
    std::string& error
);
}
