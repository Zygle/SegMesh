#include "segment_merging.h"
#include "face_graph.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace
{
using segmesh::detail::FaceGraphEdgeInfo;
using segmesh::detail::buildFaceGraphEdges;

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
            if (pair.commonLength <= 1.0e-12)
            {
                continue;
            }

            const uint32_t segmentA = pair.segmentA;
            const uint32_t segmentB = pair.segmentB;
            const double commonAverage = pair.commonDifference / pair.commonLength;
            // Eq. (14): compare the shared boundary against the union of the two current boundaries.
            const double unionDifference =
                boundaryDifference[static_cast<std::size_t>(segmentA)]
                + boundaryDifference[static_cast<std::size_t>(segmentB)]
                - pair.commonDifference;
            const double unionLength =
                boundaryLength[static_cast<std::size_t>(segmentA)]
                + boundaryLength[static_cast<std::size_t>(segmentB)]
                - pair.commonLength;
            if (unionLength <= 1.0e-12)
            {
                continue;
            }

            const double unionAverage = unionDifference / unionLength;
            if (unionAverage <= 1.0e-12)
            {
                continue;
            }

            const double relativeCost = commonAverage / unionAverage;
            if (!std::isfinite(relativeCost))
            {
                continue;
            }

            queue.emplace(relativeCost, segmentA, segmentB);
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
    std::vector<uint32_t>& inOutTriangleLabels,
    std::string& error
)
{
    error.clear();

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
    (void)averageDifference;

    for (;;)
    {
        uint32_t segmentCount = 0;
        for (const uint32_t label : inOutTriangleLabels)
        {
            segmentCount = std::max(segmentCount, label + 1);
        }

        if (segmentCount <= 1)
        {
            return true;
        }

        std::vector<uint32_t> segmentSizes(segmentCount, 0);
        for (const uint32_t label : inOutTriangleLabels)
        {
            ++segmentSizes[static_cast<std::size_t>(label)];
        }

        struct NeighborStats
        {
            double sharedDifference = 0.0;
            double sharedLength = 0.0;
        };

        std::vector<std::unordered_map<uint32_t, NeighborStats> > neighborStats(segmentCount);

        for (const FaceGraphEdgeInfo& edge : edges)
        {
            const uint32_t label0 = inOutTriangleLabels[static_cast<std::size_t>(edge.face0)];
            const uint32_t label1 = inOutTriangleLabels[static_cast<std::size_t>(edge.face1)];
            if (label0 == label1)
            {
                continue;
            }

            const double edgeBoundaryDifference = edge.edgeLength * edge.difference;
            NeighborStats& stats01 = neighborStats[static_cast<std::size_t>(label0)][label1];
            NeighborStats& stats10 = neighborStats[static_cast<std::size_t>(label1)][label0];
            stats01.sharedDifference += edgeBoundaryDifference;
            stats01.sharedLength += edge.edgeLength;
            stats10.sharedDifference += edgeBoundaryDifference;
            stats10.sharedLength += edge.edgeLength;
        }

        uint32_t segmentToMerge = segmentCount;
        uint32_t mergeTarget = segmentCount;
        uint32_t smallestSegmentSize = std::numeric_limits<uint32_t>::max();
        double bestBoundaryAverage = std::numeric_limits<double>::infinity();
        double bestSharedLength = -1.0;

        for (uint32_t segmentIndex = 0; segmentIndex < segmentCount; ++segmentIndex)
        {
            const uint32_t segmentSize = segmentSizes[static_cast<std::size_t>(segmentIndex)];
            if (segmentSize == 0 || segmentSize >= minTriangleCount)
            {
                continue;
            }

            uint32_t bestNeighbor = segmentCount;
            double bestNeighborAverage = std::numeric_limits<double>::infinity();
            double bestNeighborSharedLength = -1.0;

            for (const auto& neighborEntry : neighborStats[static_cast<std::size_t>(segmentIndex)])
            {
                const uint32_t neighborIndex = neighborEntry.first;
                if (neighborIndex == segmentIndex || segmentSizes[static_cast<std::size_t>(neighborIndex)] == 0)
                {
                    continue;
                }

                const NeighborStats& stats = neighborEntry.second;
                if (stats.sharedLength <= 1.0e-12)
                {
                    continue;
                }

                const double boundaryAverage = stats.sharedDifference / stats.sharedLength;
                if (boundaryAverage < bestNeighborAverage
                    || (std::abs(boundaryAverage - bestNeighborAverage) <= 1.0e-12
                        && stats.sharedLength > bestNeighborSharedLength))
                {
                    bestNeighbor = neighborIndex;
                    bestNeighborAverage = boundaryAverage;
                    bestNeighborSharedLength = stats.sharedLength;
                }
            }

            if (bestNeighbor == segmentCount)
            {
                continue;
            }

            if (segmentSize < smallestSegmentSize
                || (segmentSize == smallestSegmentSize && bestNeighborAverage < bestBoundaryAverage)
                || (segmentSize == smallestSegmentSize
                    && std::abs(bestNeighborAverage - bestBoundaryAverage) <= 1.0e-12
                    && bestNeighborSharedLength > bestSharedLength))
            {
                segmentToMerge = segmentIndex;
                mergeTarget = bestNeighbor;
                smallestSegmentSize = segmentSize;
                bestBoundaryAverage = bestNeighborAverage;
                bestSharedLength = bestNeighborSharedLength;
            }
        }

        if (segmentToMerge == segmentCount || mergeTarget == segmentCount)
        {
            break;
        }

        for (uint32_t& label : inOutTriangleLabels)
        {
            if (label == segmentToMerge)
            {
                label = mergeTarget;
            }
        }

        densifySegmentLabels(inOutTriangleLabels);
    }

    return true;
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
