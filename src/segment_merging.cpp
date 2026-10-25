#include "segment_merging.h"
#include "face_graph.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace
{
using segmesh::detail::FaceGraphEdgeInfo;
using segmesh::detail::buildFaceGraphEdges;

constexpr double kBoundaryEpsilon = 1.0e-12;

// Eq. (14), shared by general merging and small-region cleanup.
std::optional<double> relativeBoundaryCost(
    double commonDifference,
    double commonLength,
    double boundaryDifferenceA,
    double boundaryLengthA,
    double boundaryDifferenceB,
    double boundaryLengthB
)
{
    const double unionDifference = boundaryDifferenceA + boundaryDifferenceB - commonDifference;
    const double unionLength = boundaryLengthA + boundaryLengthB - commonLength;
    if (!std::isfinite(commonDifference) || !std::isfinite(commonLength)
        || !std::isfinite(unionDifference) || !std::isfinite(unionLength)
        || commonDifference < 0.0 || unionDifference < 0.0
        || commonLength <= kBoundaryEpsilon || unionLength <= kBoundaryEpsilon)
    {
        return std::nullopt;
    }

    const double commonAverage = commonDifference / commonLength;
    const double unionAverage = unionDifference / unionLength;
    if (!std::isfinite(commonAverage) || !std::isfinite(unionAverage))
    {
        return std::nullopt;
    }
    if (commonAverage <= kBoundaryEpsilon && unionAverage <= kBoundaryEpsilon)
    {
        return 0.0;
    }
    if (unionAverage <= kBoundaryEpsilon)
    {
        return std::nullopt;
    }

    const double cost = commonAverage / unionAverage;
    return std::isfinite(cost) && cost >= 0.0 ? std::optional<double>(cost) : std::nullopt;
}

uint32_t findSegmentRoot(std::vector<uint32_t>& parents, uint32_t segment)
{
    uint32_t root = segment;
    while (parents[static_cast<std::size_t>(root)] != root)
    {
        root = parents[static_cast<std::size_t>(root)];
    }

    while (parents[static_cast<std::size_t>(segment)] != segment)
    {
        const uint32_t parent = parents[static_cast<std::size_t>(segment)];
        parents[static_cast<std::size_t>(segment)] = root;
        segment = parent;
    }

    return root;
}

void densifySegmentLabels(std::vector<uint32_t>& inOutTriangleLabels)
{
    uint32_t labelCount = 0;
    for (const uint32_t label : inOutTriangleLabels)
    {
        labelCount = std::max(labelCount, label + 1);
    }

    if (labelCount == 0)
    {
        return;
    }

    std::vector<int32_t> denseLabels(labelCount, -1);
    uint32_t nextDenseLabel = 0;
    for (uint32_t& label : inOutTriangleLabels)
    {
        int32_t& denseLabel = denseLabels[static_cast<std::size_t>(label)];
        if (denseLabel < 0)
        {
            denseLabel = static_cast<int32_t>(nextDenseLabel);
            ++nextDenseLabel;
        }

        label = static_cast<uint32_t>(denseLabel);
    }
}
}

namespace segmesh
{
bool mergeSegmentsByBoundaryCost(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t targetSegmentCount,
    double maxRelativeMergeCost,
    std::vector<uint32_t>& inOutTriangleLabels,
    std::string& error
)
{
    error.clear();

    const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (faceCount == 0)
    {
        error = "Cannot merge segments on an empty mesh.";
        return false;
    }

    if (inOutTriangleLabels.size() != faceCount)
    {
        error = "Segment label count does not match the face count.";
        return false;
    }

    if (targetSegmentCount == 0)
    {
        error = "Merged segmentation must keep at least one segment.";
        return false;
    }

    if (!(maxRelativeMergeCost >= 0.0))
    {
        error = "Merge cost limit must be non-negative.";
        return false;
    }

    std::vector<FaceGraphEdgeInfo> edges;
    double averageDifference = 0.0;
    if (!buildFaceGraphEdges(mesh, segmentationModelType, edges, averageDifference, error))
    {
        return false;
    }
    (void)averageDifference;

    uint32_t initialSegmentCount = 0;
    for (const uint32_t label : inOutTriangleLabels)
    {
        initialSegmentCount = std::max(initialSegmentCount, label + 1);
    }

    if (initialSegmentCount == 0)
    {
        error = "Segmentation labels are empty.";
        return false;
    }

    for (const uint32_t label : inOutTriangleLabels)
    {
        if (label >= initialSegmentCount)
        {
            error = "Segmentation labels contain an out-of-range segment id.";
            return false;
        }
    }

    std::vector<uint32_t> parents(initialSegmentCount, 0);
    for (uint32_t segmentIndex = 0; segmentIndex < initialSegmentCount; ++segmentIndex)
    {
        parents[static_cast<std::size_t>(segmentIndex)] = segmentIndex;
    }

    std::vector<bool> labelSeen(initialSegmentCount, false);
    uint32_t currentSegmentCount = 0;
    for (const uint32_t label : inOutTriangleLabels)
    {
        if (!labelSeen[static_cast<std::size_t>(label)])
        {
            labelSeen[static_cast<std::size_t>(label)] = true;
            ++currentSegmentCount;
        }
    }

    while (currentSegmentCount > targetSegmentCount)
    {
        std::vector<double> boundaryDifference(initialSegmentCount, 0.0);
        std::vector<double> boundaryLength(initialSegmentCount, 0.0);

        struct AdjacentSegmentPair
        {
            uint32_t segmentA = 0;
            uint32_t segmentB = 0;
            double commonDifference = 0.0;
            double commonLength = 0.0;
        };

        const auto pairKey = [](uint32_t a, uint32_t b) -> uint64_t
        {
            return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
        };

        std::unordered_map<uint64_t, AdjacentSegmentPair> adjacentPairs;
        adjacentPairs.reserve(edges.size());

        for (const FaceGraphEdgeInfo& edge : edges)
        {
            const uint32_t label0 = inOutTriangleLabels[static_cast<std::size_t>(edge.face0)];
            const uint32_t label1 = inOutTriangleLabels[static_cast<std::size_t>(edge.face1)];
            const uint32_t root0 = findSegmentRoot(parents, label0);
            const uint32_t root1 = findSegmentRoot(parents, label1);

            if (root0 == root1)
            {
                continue;
            }

            const double edgeBoundaryDifference = edge.edgeLength * edge.difference;
            const double edgeBoundaryLength = edge.edgeLength;

            boundaryDifference[static_cast<std::size_t>(root0)] += edgeBoundaryDifference;
            boundaryDifference[static_cast<std::size_t>(root1)] += edgeBoundaryDifference;
            boundaryLength[static_cast<std::size_t>(root0)] += edgeBoundaryLength;
            boundaryLength[static_cast<std::size_t>(root1)] += edgeBoundaryLength;

            const uint32_t a = std::min(root0, root1);
            const uint32_t b = std::max(root0, root1);
            auto [pairIt, inserted] = adjacentPairs.try_emplace(pairKey(a, b));
            if (inserted)
            {
                pairIt->second.segmentA = a;
                pairIt->second.segmentB = b;
            }

            pairIt->second.commonDifference += edgeBoundaryDifference;
            pairIt->second.commonLength += edgeBoundaryLength;
        }

        using MergeCandidate = std::tuple<double, uint32_t, uint32_t>;
        std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, std::greater<MergeCandidate> > queue;

        for (const auto& pairEntry : adjacentPairs)
        {
            const AdjacentSegmentPair& pair = pairEntry.second;
            const uint32_t segmentA = pair.segmentA;
            const uint32_t segmentB = pair.segmentB;
            const auto relativeCost = relativeBoundaryCost(
                pair.commonDifference,
                pair.commonLength,
                boundaryDifference[segmentA],
                boundaryLength[segmentA],
                boundaryDifference[segmentB],
                boundaryLength[segmentB]
            );
            if (!relativeCost.has_value())
            {
                continue;
            }

            queue.emplace(*relativeCost, segmentA, segmentB);
        }

        if (queue.empty())
        {
            break;
        }

        const MergeCandidate bestMerge = queue.top();
        const double bestCost = std::get<0>(bestMerge);
        const uint32_t segmentA = std::get<1>(bestMerge);
        const uint32_t segmentB = std::get<2>(bestMerge);
        if (bestCost > maxRelativeMergeCost)
        {
            break;
        }

        const uint32_t keepRoot = std::min(segmentA, segmentB);
        const uint32_t mergeRoot = std::max(segmentA, segmentB);
        parents[static_cast<std::size_t>(mergeRoot)] = keepRoot;
        --currentSegmentCount;
    }

    std::vector<int32_t> denseLabels(initialSegmentCount, -1);
    uint32_t nextDenseLabel = 0;
    for (uint32_t& label : inOutTriangleLabels)
    {
        const uint32_t root = findSegmentRoot(parents, label);
        int32_t& denseLabel = denseLabels[static_cast<std::size_t>(root)];
        if (denseLabel < 0)
        {
            denseLabel = static_cast<int32_t>(nextDenseLabel);
            ++nextDenseLabel;
        }

        label = static_cast<uint32_t>(denseLabel);
    }

    return true;
}

bool mergeSmallSegmentsByTriangleCount(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t minTriangleCount,
    uint32_t minimumSegmentCount,
    double maxRelativeMergeCost,
    std::vector<uint32_t>& inOutTriangleLabels,
    std::string& error
)
{
    error.clear();

    if (minimumSegmentCount == 0)
    {
        error = "Small-segment cleanup must keep at least one segment.";
        return false;
    }
    if (!(maxRelativeMergeCost >= 0.0))
    {
        error = "Cleanup cost limit must be non-negative.";
        return false;
    }
    if (minTriangleCount <= 1)
    {
        return true;
    }

    const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (faceCount == 0)
    {
        error = "Cannot clean up small segments on an empty mesh.";
        return false;
    }
    if (inOutTriangleLabels.size() != faceCount)
    {
        error = "Segment label count does not match the face count.";
        return false;
    }

    std::vector<FaceGraphEdgeInfo> edges;
    double averageDifference = 0.0;
    if (!buildFaceGraphEdges(mesh, segmentationModelType, edges, averageDifference, error))
    {
        return false;
    }

    // Count actual regions even when earlier operations left gaps in label IDs.
    densifySegmentLabels(inOutTriangleLabels);
    for (;;)
    {
        uint32_t segmentCount = 0;
        for (const uint32_t label : inOutTriangleLabels)
        {
            segmentCount = std::max(segmentCount, label + 1);
        }
        if (segmentCount <= minimumSegmentCount)
        {
            return true;
        }

        std::vector<uint32_t> segmentSizes(segmentCount, 0);
        for (const uint32_t label : inOutTriangleLabels)
        {
            ++segmentSizes[label];
        }

        struct NeighborStats
        {
            double sharedDifference = 0.0;
            double sharedLength = 0.0;
        };

        std::vector<std::unordered_map<uint32_t, NeighborStats> > neighborStats(segmentCount);
        std::vector<double> boundaryDifference(segmentCount, 0.0);
        std::vector<double> boundaryLength(segmentCount, 0.0);
        for (const FaceGraphEdgeInfo& edge : edges)
        {
            const uint32_t label0 = inOutTriangleLabels[edge.face0];
            const uint32_t label1 = inOutTriangleLabels[edge.face1];
            if (label0 == label1)
            {
                continue;
            }

            const double difference = edge.edgeLength * edge.difference;
            boundaryDifference[label0] += difference;
            boundaryDifference[label1] += difference;
            boundaryLength[label0] += edge.edgeLength;
            boundaryLength[label1] += edge.edgeLength;
            NeighborStats& stats01 = neighborStats[label0][label1];
            NeighborStats& stats10 = neighborStats[label1][label0];
            stats01.sharedDifference += difference;
            stats01.sharedLength += edge.edgeLength;
            stats10.sharedDifference += difference;
            stats10.sharedLength += edge.edgeLength;
        }

        // Smallest region first, then cheapest boundary, longest shared boundary,
        // and finally label IDs so unordered-map iteration cannot break ties.
        using Candidate = std::tuple<uint32_t, double, double, uint32_t, uint32_t>;
        std::optional<Candidate> best;
        for (uint32_t source = 0; source < segmentCount; ++source)
        {
            if (segmentSizes[source] == 0 || segmentSizes[source] >= minTriangleCount)
            {
                continue;
            }
            for (const auto& [target, stats] : neighborStats[source])
            {
                const auto cost = relativeBoundaryCost(
                    stats.sharedDifference,
                    stats.sharedLength,
                    boundaryDifference[source],
                    boundaryLength[source],
                    boundaryDifference[target],
                    boundaryLength[target]
                );
                if (!cost.has_value() || *cost > maxRelativeMergeCost)
                {
                    continue;
                }
                const Candidate candidate{segmentSizes[source], *cost, -stats.sharedLength, source, target};
                if (!best.has_value() || candidate < *best)
                {
                    best = candidate;
                }
            }
        }
        if (!best.has_value())
        {
            return true;
        }

        const uint32_t source = std::get<3>(*best);
        const uint32_t target = std::get<4>(*best);
        for (uint32_t& label : inOutTriangleLabels)
        {
            if (label == source)
            {
                label = target;
            }
        }
        densifySegmentLabels(inOutTriangleLabels);
    }
}

bool mergeAdjacentSegmentsByTriangles(
    const CpuMesh& mesh,
    uint32_t sourceTriangle,
    uint32_t targetTriangle,
    std::vector<uint32_t>& inOutTriangleLabels,
    std::string& error
)
{
    error.clear();

    const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (faceCount == 0)
    {
        error = "Cannot merge segments on an empty mesh.";
        return false;
    }
    if (mesh.faceAdjacency.size() != faceCount)
    {
        error = "Mesh adjacency data is incomplete.";
        return false;
    }
    if (inOutTriangleLabels.size() != faceCount)
    {
        error = "Segment label count does not match the face count.";
        return false;
    }
    if (sourceTriangle >= faceCount || targetTriangle >= faceCount)
    {
        error = "Selected triangle index is out of range.";
        return false;
    }

    const uint32_t sourceLabel = inOutTriangleLabels[static_cast<std::size_t>(sourceTriangle)];
    const uint32_t targetLabel = inOutTriangleLabels[static_cast<std::size_t>(targetTriangle)];
    if (sourceLabel == targetLabel)
    {
        error = "Select a different neighboring segment.";
        return false;
    }

    bool adjacent = false;
    for (uint32_t faceIndex = 0; faceIndex < faceCount && !adjacent; ++faceIndex)
    {
        if (inOutTriangleLabels[static_cast<std::size_t>(faceIndex)] != sourceLabel)
        {
            continue;
        }

        const FaceAdjacency& faceAdjacency = mesh.faceAdjacency[static_cast<std::size_t>(faceIndex)];
        for (const int32_t signedNeighbor : faceAdjacency.neighbors)
        {
            if (signedNeighbor < 0)
            {
                continue;
            }
            const uint32_t neighbor = static_cast<uint32_t>(signedNeighbor);
            if (neighbor >= faceCount)
            {
                error = "Mesh adjacency contains an out-of-range neighboring face.";
                return false;
            }
            if (inOutTriangleLabels[static_cast<std::size_t>(neighbor)] == targetLabel)
            {
                adjacent = true;
                break;
            }
        }
    }

    if (!adjacent)
    {
        error = "The selected segments are not neighbors.";
        return false;
    }

    for (uint32_t& label : inOutTriangleLabels)
    {
        if (label == sourceLabel)
        {
            label = targetLabel;
        }
    }
    return true;
}
}
