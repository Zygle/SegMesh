#include "border_curves.h"

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
void BorderCurveCache::invalidate()
{
    traceValid_ = false;
    fitValid_ = false;
    triangleLabels_.clear();
    tracedChains_.clear();
    fittedChains_.clear();
    boundaryEdgeCount_ = 0;
    skippedChainCount_ = 0;
    cubicPieceCount_ = 0;
    maxFitError_ = 0.0;
    rmsFitError_ = 0.0;
}

bool BorderCurveCache::buildOverlay(
    const CpuMesh& mesh,
    uint64_t meshGeneration,
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

    if (!traceValid_ || meshGeneration_ != meshGeneration
        || !std::equal(
            triangleLabels.begin(), triangleLabels.end(),
            triangleLabels_.begin(), triangleLabels_.end()
        ))
    {
        invalidate();
        if (!detail::traceSegmentBorders(
                mesh, triangleLabels, tracedChains_, boundaryEdgeCount_, error
            ))
        {
            return false;
        }
        meshGeneration_ = meshGeneration;
        triangleLabels_.assign(triangleLabels.begin(), triangleLabels.end());
        traceValid_ = true;
    }

    if (!fitValid_ || fitTolerance_ != settings.fitTolerance)
    {
        fitValid_ = false;
        if (!fitTracedBorders(mesh, settings.fitTolerance, error))
        {
            return false;
        }
        fitTolerance_ = settings.fitTolerance;
        fitValid_ = true;
    }

    return sampleFittedBorders(settings.drawingTolerance, outOverlay, error);
}

bool BorderCurveCache::fitTracedBorders(
    const CpuMesh& mesh,
    double fitTolerance,
    std::string& error
)
{
    std::vector<FittedChain> fittedChains;
    uint32_t skippedChainCount = 0;
    uint32_t cubicPieceCount = 0;
    double maximumFitError = 0.0;
    double totalSquaredFitError = 0.0;
    uint64_t totalFittedPointCount = 0;
    for (const detail::TracedChain& chain : tracedChains_)
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
            ++skippedChainCount;
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
                fitTolerance,
                chainStartTangent,
                chainEndTangent,
                pieces,
                error))
        {
            return false;
        }
        if (pieces.empty())
        {
            ++skippedChainCount;
            continue;
        }

        double chainMaximumFitError = 0.0;
        double chainSquaredFitError = 0.0;
        for (const detail::CubicPiece& piece : pieces)
        {
            chainMaximumFitError = std::max(
                chainMaximumFitError,
                piece.fitStats.maxError
            );
            const std::size_t piecePointCount = piece.lastSample - piece.firstSample + 1;
            chainSquaredFitError += piece.fitStats.rmsError * piece.fitStats.rmsError
                * static_cast<double>(piecePointCount);
        }

        if (pieces.size() > std::numeric_limits<uint32_t>::max()
            || cubicPieceCount > std::numeric_limits<uint32_t>::max() - pieces.size())
        {
            error = "The generated border overlay exceeds 32-bit statistics limits.";
            return false;
        }
        cubicPieceCount += static_cast<uint32_t>(pieces.size());
        maximumFitError = std::max(maximumFitError, chainMaximumFitError);
        totalSquaredFitError += chainSquaredFitError;
        totalFittedPointCount += fittedPoints.size();
        const double chainRmsFitError = std::sqrt(
            chainSquaredFitError / static_cast<double>(fittedPoints.size())
        );
        BorderCurveChainStats stats{};
        stats.firstLabel = chain.labels.first;
        stats.secondLabel = chain.labels.second;
        stats.sourcePointCount = static_cast<uint32_t>(sourcePoints.size());
        stats.fittedPointCount = static_cast<uint32_t>(fittedPoints.size());
        stats.cubicPieceCount = static_cast<uint32_t>(pieces.size());
        stats.maxFitError = chainMaximumFitError;
        stats.rmsFitError = chainRmsFitError;
        stats.closedLoopArc = chain.closedLoopArc;
        fittedChains.push_back({std::move(pieces), stats});
    }

    // Publish only a complete fit; drawing failures can retry sampling this result.
    fittedChains_ = std::move(fittedChains);
    skippedChainCount_ = skippedChainCount;
    cubicPieceCount_ = cubicPieceCount;
    maxFitError_ = maximumFitError;
    rmsFitError_ = totalFittedPointCount > 0
        ? std::sqrt(totalSquaredFitError / static_cast<double>(totalFittedPointCount))
        : 0.0;
    return true;
}

bool BorderCurveCache::sampleFittedBorders(
    double drawingTolerance,
    BorderCurveOverlay& outOverlay,
    std::string& error
) const
{
    outOverlay.boundaryEdgeCount = boundaryEdgeCount_;
    outOverlay.skippedChainCount = skippedChainCount_;
    outOverlay.cubicPieceCount = cubicPieceCount_;
    outOverlay.maxFitError = maxFitError_;
    outOverlay.rmsFitError = rmsFitError_;

    for (const FittedChain& chain : fittedChains_)
    {
        std::vector<Float3> renderedPoints;
        bool drawingSubdivisionCapped = false;
        for (const detail::CubicPiece& piece : chain.pieces)
        {
            bool pieceSubdivisionCapped = false;
            std::vector<Float3> flattened = detail::flattenCubic(
                piece.controlPoints,
                drawingTolerance,
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
        }
        if (!appendPolyline(renderedPoints, outOverlay.vertices, outOverlay.indices, error))
        {
            return false;
        }
        if (renderedPoints.size() > std::numeric_limits<uint32_t>::max())
        {
            error = "The generated border overlay exceeds 32-bit statistics limits.";
            return false;
        }

        BorderCurveChainStats stats = chain.stats;
        stats.renderedPointCount = static_cast<uint32_t>(renderedPoints.size());
        stats.drawingSubdivisionCapped = drawingSubdivisionCapped;
        outOverlay.chains.push_back(stats);
        if (drawingSubdivisionCapped)
        {
            ++outOverlay.drawingCappedChainCount;
        }
    }
    return true;
}

bool buildSegmentBorderOverlay(
    const CpuMesh& mesh,
    std::span<const uint32_t> triangleLabels,
    const BorderCurveSettings& settings,
    BorderCurveOverlay& outOverlay,
    std::string& error
)
{
    BorderCurveCache cache;
    return cache.buildOverlay(mesh, 0, triangleLabels, settings, outOverlay, error);
}
}
