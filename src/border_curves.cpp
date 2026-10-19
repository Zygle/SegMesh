#include "border_curves.h"
#include "border_trace.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{
constexpr double kCoincidentPointEpsilon = 1.0e-12;
constexpr uint32_t kMaximumDrawingSubdivisionDepth = 12;

void flattenCubicRecursive(
    const std::array<segmesh::Float3, 4>& cubic,
    double tolerance,
    uint32_t depth,
    std::vector<segmesh::Float3>& outPoints,
    bool& outSubdivisionCapped
)
{
    const double flatness = segmesh::detail::cubicFlatness(cubic);
    if (flatness <= tolerance || depth == kMaximumDrawingSubdivisionDepth)
    {
        if (flatness > tolerance)
        {
            outSubdivisionCapped = true;
        }
        outPoints.push_back(cubic.back());
        return;
    }

    std::array<segmesh::Float3, 4> left{};
    std::array<segmesh::Float3, 4> right{};
    segmesh::detail::splitCubicAtHalf(cubic, left, right);
    flattenCubicRecursive(left, tolerance, depth + 1, outPoints, outSubdivisionCapped);
    flattenCubicRecursive(right, tolerance, depth + 1, outPoints, outSubdivisionCapped);
}

std::vector<segmesh::Float3> flattenCubic(
    const std::array<segmesh::Float3, 4>& cubic,
    double tolerance,
    bool& outSubdivisionCapped
)
{
    std::vector<segmesh::Float3> points;
    points.push_back(cubic.front());
    flattenCubicRecursive(cubic, tolerance, 0, points, outSubdivisionCapped);
    return points;
}

bool appendPolyline(
    std::span<const segmesh::Float3> points,
    std::vector<segmesh::Float3>& outVertices,
    std::vector<uint32_t>& outIndices,
    std::string& error
)
{
    if (points.size() < 2)
    {
        return true;
    }
    if (points.size() > std::numeric_limits<uint32_t>::max()
        || outVertices.size() > std::numeric_limits<uint32_t>::max() - points.size())
    {
        error = "The generated border overlay exceeds the 32-bit vertex index limit.";
        return false;
    }

    const uint32_t firstVertex = static_cast<uint32_t>(outVertices.size());
    outVertices.insert(outVertices.end(), points.begin(), points.end());
    for (std::size_t index = 1; index < points.size(); ++index)
    {
        outIndices.push_back(firstVertex + static_cast<uint32_t>(index - 1));
        outIndices.push_back(firstVertex + static_cast<uint32_t>(index));
    }
    return true;
}
}

namespace segmesh
{
bool buildSegmentBorderOverlay(
    const CpuMesh& mesh,
    std::span<const uint32_t> triangleLabels,
    const BorderCurveSettings& settings,
    BorderCurveOverlay& outOverlay,
    std::string& error
)
{
    outOverlay = {};
    error.clear();

    const std::size_t faceCount = mesh.faceAdjacency.size();
    if (triangleLabels.size() != faceCount)
    {
        error = "Segment label count does not match the mesh face count.";
        return false;
    }
    if (mesh.faceRenderEdges.size() != faceCount
        || mesh.faceTopologyVertexIndices.size() != faceCount
        || faceCount > std::numeric_limits<std::size_t>::max() / 3
        || mesh.indices.size() != faceCount * 3)
    {
        error = "Mesh edge topology data is incomplete.";
        return false;
    }
    if (!std::isfinite(settings.fitTolerance) || settings.fitTolerance < 0.0)
    {
        error = "Border-curve fit tolerance must be a finite non-negative value.";
        return false;
    }
    if (!std::isfinite(settings.drawingTolerance) || settings.drawingTolerance < 0.0)
    {
        error = "Border-curve drawing tolerance must be a finite non-negative value.";
        return false;
    }

    std::vector<detail::TracedChain> tracedChains;
    if (!detail::traceSegmentBorders(
            mesh,
            triangleLabels,
            tracedChains,
            outOverlay.boundaryEdgeCount,
            error
        ))
    {
        return false;
    }

    double totalSquaredFitError = 0.0;
    uint64_t totalFittedPointCount = 0;
    for (const detail::TracedChain& chain : tracedChains)
    {
        std::vector<Float3> sourcePoints;
        sourcePoints.reserve(chain.vertexIndices.size());
        for (const uint32_t vertexIndex : chain.vertexIndices)
        {
            if (vertexIndex >= mesh.canonicalPositions.size())
            {
                error = "A traced border chain contains an out-of-range canonical vertex.";
                return false;
            }
            sourcePoints.push_back(mesh.canonicalPositions[vertexIndex]);
        }

        std::vector<Float3> fittedPoints = detail::removeConsecutiveCoincidentPoints(sourcePoints);
        if (fittedPoints.size() < 2)
        {
            ++outOverlay.skippedChainCount;
            continue;
        }

        Eigen::Vector3d chainStartTangent = detail::forwardTangentAt(fittedPoints, 0);
        Eigen::Vector3d chainEndTangent = detail::forwardTangentAt(
            fittedPoints,
            fittedPoints.size() - 1
        );
        if (chain.hasEndpointTangents
            && chain.startTangent.allFinite()
            && chain.endTangent.allFinite()
            && chain.startTangent.squaredNorm()
                > kCoincidentPointEpsilon * kCoincidentPointEpsilon
            && chain.endTangent.squaredNorm()
                > kCoincidentPointEpsilon * kCoincidentPointEpsilon)
        {
            chainStartTangent = chain.startTangent.normalized();
            chainEndTangent = chain.endTangent.normalized();
        }

        std::vector<detail::CubicPiece> pieces;
        if (!detail::fitAdaptiveCubicPieces(
                fittedPoints,
                settings.fitTolerance,
                chainStartTangent,
                chainEndTangent,
                pieces,
                error))
        {
            return false;
        }
        if (pieces.empty())
        {
            ++outOverlay.skippedChainCount;
            continue;
        }

        std::vector<Float3> renderedPoints;
        bool drawingSubdivisionCapped = false;
        double chainMaximumFitError = 0.0;
        double chainSquaredFitError = 0.0;
        for (const detail::CubicPiece& piece : pieces)
        {
            bool pieceSubdivisionCapped = false;
            std::vector<Float3> flattened = flattenCubic(
                piece.controlPoints,
                settings.drawingTolerance,
                pieceSubdivisionCapped
            );
            drawingSubdivisionCapped |= pieceSubdivisionCapped;
            if (renderedPoints.empty())
            {
                renderedPoints = std::move(flattened);
            }
            else
            {
                renderedPoints.insert(renderedPoints.end(), flattened.begin() + 1, flattened.end());
            }

            if (!appendPolyline(
                    piece.controlPoints,
                    outOverlay.controlVertices,
                    outOverlay.controlIndices,
                    error))
            {
                return false;
            }
            chainMaximumFitError = std::max(
                chainMaximumFitError,
                piece.fitStats.maxError
            );
            const std::size_t piecePointCount = piece.lastSample - piece.firstSample + 1;
            chainSquaredFitError += piece.fitStats.rmsError * piece.fitStats.rmsError
                * static_cast<double>(piecePointCount);
        }
        if (!appendPolyline(renderedPoints, outOverlay.vertices, outOverlay.indices, error))
        {
            return false;
        }

        if (pieces.size() > std::numeric_limits<uint32_t>::max()
            || renderedPoints.size() > std::numeric_limits<uint32_t>::max()
            || outOverlay.cubicPieceCount > std::numeric_limits<uint32_t>::max() - pieces.size())
        {
            error = "The generated border overlay exceeds 32-bit statistics limits.";
            return false;
        }
        outOverlay.cubicPieceCount += static_cast<uint32_t>(pieces.size());
        if (drawingSubdivisionCapped)
        {
            ++outOverlay.drawingCappedChainCount;
        }
        outOverlay.maxFitError = std::max(outOverlay.maxFitError, chainMaximumFitError);
        totalSquaredFitError += chainSquaredFitError;
        totalFittedPointCount += fittedPoints.size();
        const double chainRmsFitError = std::sqrt(
            chainSquaredFitError / static_cast<double>(fittedPoints.size())
        );
        outOverlay.chains.push_back({
            chain.labels.first,
            chain.labels.second,
            static_cast<uint32_t>(sourcePoints.size()),
            static_cast<uint32_t>(fittedPoints.size()),
            static_cast<uint32_t>(pieces.size()),
            static_cast<uint32_t>(renderedPoints.size()),
            chainMaximumFitError,
            chainRmsFitError,
            drawingSubdivisionCapped,
            chain.closedLoopArc,
        });
    }

    if (totalFittedPointCount > 0)
    {
        outOverlay.rmsFitError = std::sqrt(
            totalSquaredFitError / static_cast<double>(totalFittedPointCount)
        );
    }
    return true;
}
}
