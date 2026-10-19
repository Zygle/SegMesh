#include "border_trace.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace segmesh::detail
{
namespace
{
constexpr double kCoincidentPointEpsilon = 1.0e-12;

struct TopologyEdge
{
    uint32_t first = 0;
    uint32_t second = 0;

    auto operator<=>(const TopologyEdge&) const = default;
};

Eigen::Vector3d toEigen(const segmesh::Float3& point)
{
    return {
        static_cast<double>(point.x),
        static_cast<double>(point.y),
        static_cast<double>(point.z),
    };
}

bool isFinite(const segmesh::Float3& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

double pointDistance(const segmesh::Float3& first, const segmesh::Float3& second)
{
    return (toEigen(first) - toEigen(second)).norm();
}

uint32_t otherEndpoint(const TopologyEdge& edge, uint32_t endpoint)
{
    return edge.first == endpoint ? edge.second : edge.first;
}

bool topologyEdgeForFaceSlot(
    const segmesh::CpuMesh& mesh,
    std::size_t faceIndex,
    std::size_t edgeSlot,
    TopologyEdge& outEdge,
    std::string& error
)
{
    const uint64_t firstRenderIndex = static_cast<uint64_t>(faceIndex) * 3;
    const auto& renderEdge = mesh.faceRenderEdges[faceIndex].indices[edgeSlot];
    std::array<uint32_t, 2> topologyVertices{};

    for (std::size_t endpoint = 0; endpoint < renderEdge.size(); ++endpoint)
    {
        const uint64_t renderIndex = renderEdge[endpoint];
        if (renderIndex < firstRenderIndex || renderIndex >= firstRenderIndex + 3)
        {
            error = "A face edge does not reference one of its three render vertices.";
            return false;
        }

        const std::size_t corner = static_cast<std::size_t>(renderIndex - firstRenderIndex);
        topologyVertices[endpoint] = mesh.faceTopologyVertexIndices[faceIndex][corner];
        if (topologyVertices[endpoint] >= mesh.canonicalPositions.size())
        {
            error = "A face edge contains an out-of-range canonical vertex index.";
            return false;
        }
    }

    if (topologyVertices[0] == topologyVertices[1])
    {
        error = "A face edge collapses to one canonical vertex.";
        return false;
    }

    outEdge = {
        std::min(topologyVertices[0], topologyVertices[1]),
        std::max(topologyVertices[0], topologyVertices[1]),
    };
    return true;
}

bool faceContainsEdge(
    const std::array<uint32_t, 3>& faceVertices,
    const TopologyEdge& edge
)
{
    const bool containsFirst = std::find(faceVertices.begin(), faceVertices.end(), edge.first)
        != faceVertices.end();
    const bool containsSecond = std::find(faceVertices.begin(), faceVertices.end(), edge.second)
        != faceVertices.end();
    return containsFirst && containsSecond;
}

std::vector<uint32_t> traceFromEdge(
    uint32_t startVertex,
    std::size_t firstEdge,
    const std::vector<TopologyEdge>& edges,
    const std::map<uint32_t, std::vector<std::size_t> >& incidentEdges,
    const std::set<uint32_t>& forcedSplitVertices,
    std::vector<bool>& visitedEdges
)
{
    std::vector<uint32_t> chain;
    chain.push_back(startVertex);

    uint32_t currentVertex = startVertex;
    std::size_t currentEdge = firstEdge;
    while (!visitedEdges[currentEdge])
    {
        visitedEdges[currentEdge] = true;
        const uint32_t nextVertex = otherEndpoint(edges[currentEdge], currentVertex);
        chain.push_back(nextVertex);

        const auto incidentIt = incidentEdges.find(nextVertex);
        if (incidentIt == incidentEdges.end()
            || incidentIt->second.size() != 2
            || forcedSplitVertices.contains(nextVertex))
        {
            break;
        }

        std::size_t nextEdge = edges.size();
        for (const std::size_t candidateEdge : incidentIt->second)
        {
            if (!visitedEdges[candidateEdge])
            {
                nextEdge = candidateEdge;
                break;
            }
        }
        if (nextEdge == edges.size())
        {
            break;
        }

        currentVertex = nextVertex;
        currentEdge = nextEdge;
    }
    return chain;
}

std::vector<uint32_t> traceCycle(
    uint32_t startVertex,
    const std::vector<TopologyEdge>& edges,
    const std::map<uint32_t, std::vector<std::size_t> >& incidentEdges,
    std::vector<bool>& visitedEdges
)
{
    std::vector<uint32_t> cycle;
    cycle.push_back(startVertex);

    const std::vector<std::size_t>& startEdges = incidentEdges.at(startVertex);
    std::size_t currentEdge = edges.size();
    uint32_t smallestNeighbor = std::numeric_limits<uint32_t>::max();
    for (const std::size_t candidateEdge : startEdges)
    {
        if (visitedEdges[candidateEdge])
        {
            continue;
        }
        const uint32_t neighbor = otherEndpoint(edges[candidateEdge], startVertex);
        if (neighbor < smallestNeighbor)
        {
            smallestNeighbor = neighbor;
            currentEdge = candidateEdge;
        }
    }

    if (currentEdge == edges.size())
    {
        return cycle;
    }

    uint32_t currentVertex = startVertex;
    while (!visitedEdges[currentEdge])
    {
        visitedEdges[currentEdge] = true;
        const uint32_t nextVertex = otherEndpoint(edges[currentEdge], currentVertex);
        if (nextVertex == startVertex)
        {
            break;
        }
        cycle.push_back(nextVertex);

        const auto incidentIt = incidentEdges.find(nextVertex);
        if (incidentIt == incidentEdges.end())
        {
            break;
        }

        std::size_t nextEdge = edges.size();
        for (const std::size_t candidateEdge : incidentIt->second)
        {
            if (!visitedEdges[candidateEdge])
            {
                nextEdge = candidateEdge;
                break;
            }
        }
        if (nextEdge == edges.size())
        {
            break;
        }

        currentVertex = nextVertex;
        currentEdge = nextEdge;
    }
    return cycle;
}

std::size_t halfPerimeterVertex(
    const std::vector<uint32_t>& cycle,
    std::span<const segmesh::Float3> positions
)
{
    std::vector<double> cumulativeLengths(cycle.size(), 0.0);
    for (std::size_t index = 1; index < cycle.size(); ++index)
    {
        cumulativeLengths[index] = cumulativeLengths[index - 1]
            + pointDistance(positions[cycle[index - 1]], positions[cycle[index]]);
    }
    const double perimeter = cumulativeLengths.back()
        + pointDistance(positions[cycle.back()], positions[cycle.front()]);
    const double targetLength = perimeter * 0.5;

    std::size_t bestIndex = 1;
    double bestDifference = std::abs(cumulativeLengths[bestIndex] - targetLength);
    for (std::size_t index = 2; index < cycle.size(); ++index)
    {
        const double difference = std::abs(cumulativeLengths[index] - targetLength);
        if (difference < bestDifference)
        {
            bestDifference = difference;
            bestIndex = index;
        }
    }
    return bestIndex;
}

Eigen::Vector3d cyclicForwardTangent(
    const std::vector<uint32_t>& cycle,
    std::size_t index,
    std::span<const segmesh::Float3> positions
)
{
    const Eigen::Vector3d current = toEigen(positions[cycle[index]]);
    Eigen::Vector3d incoming{};
    Eigen::Vector3d outgoing{};

    for (std::size_t offset = 1; offset < cycle.size(); ++offset)
    {
        const std::size_t previousIndex = (index + cycle.size() - offset) % cycle.size();
        incoming = current - toEigen(positions[cycle[previousIndex]]);
        if (incoming.squaredNorm() > kCoincidentPointEpsilon * kCoincidentPointEpsilon)
        {
            incoming.normalize();
            break;
        }
    }
    for (std::size_t offset = 1; offset < cycle.size(); ++offset)
    {
        const std::size_t nextIndex = (index + offset) % cycle.size();
        outgoing = toEigen(positions[cycle[nextIndex]]) - current;
        if (outgoing.squaredNorm() > kCoincidentPointEpsilon * kCoincidentPointEpsilon)
        {
            outgoing.normalize();
            break;
        }
    }

    Eigen::Vector3d tangent = incoming + outgoing;
    if (tangent.squaredNorm() <= kCoincidentPointEpsilon * kCoincidentPointEpsilon)
    {
        tangent = outgoing.squaredNorm() > kCoincidentPointEpsilon * kCoincidentPointEpsilon
            ? outgoing
            : incoming;
    }
    if (tangent.squaredNorm() <= kCoincidentPointEpsilon * kCoincidentPointEpsilon)
    {
        return {};
    }
    return tangent.normalized();
}

void traceLabelPairEdges(
    const LabelPair& labels,
    const std::set<TopologyEdge>& edgeSet,
    std::span<const segmesh::Float3> positions,
    const std::set<uint32_t>& forcedSplitVertices,
    std::vector<TracedChain>& outChains
)
{
    const std::vector<TopologyEdge> edges(edgeSet.begin(), edgeSet.end());
    std::map<uint32_t, std::vector<std::size_t> > incidentEdges;
    for (std::size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex)
    {
        incidentEdges[edges[edgeIndex].first].push_back(edgeIndex);
        incidentEdges[edges[edgeIndex].second].push_back(edgeIndex);
    }
    for (auto& [vertex, vertexEdges] : incidentEdges)
    {
        std::sort(
            vertexEdges.begin(),
            vertexEdges.end(),
            [&](std::size_t first, std::size_t second)
            {
                return otherEndpoint(edges[first], vertex) < otherEndpoint(edges[second], vertex);
            }
        );
    }

    std::vector<bool> visitedEdges(edges.size(), false);
    for (const auto& [vertex, vertexEdges] : incidentEdges)
    {
        if (vertexEdges.size() == 2 && !forcedSplitVertices.contains(vertex))
        {
            continue;
        }

        for (const std::size_t edgeIndex : vertexEdges)
        {
            if (!visitedEdges[edgeIndex])
            {
                outChains.push_back({
                    labels,
                    traceFromEdge(
                        vertex,
                        edgeIndex,
                        edges,
                        incidentEdges,
                        forcedSplitVertices,
                        visitedEdges
                    ),
                    false,
                });
            }
        }
    }

    while (true)
    {
        uint32_t cycleStart = std::numeric_limits<uint32_t>::max();
        for (const auto& [vertex, vertexEdges] : incidentEdges)
        {
            const bool hasUnvisitedEdge = std::any_of(
                vertexEdges.begin(),
                vertexEdges.end(),
                [&](std::size_t edgeIndex) { return !visitedEdges[edgeIndex]; }
            );
            if (hasUnvisitedEdge)
            {
                cycleStart = vertex;
                break;
            }
        }
        if (cycleStart == std::numeric_limits<uint32_t>::max())
        {
            break;
        }

        std::vector<uint32_t> cycle = traceCycle(
            cycleStart,
            edges,
            incidentEdges,
            visitedEdges
        );
        if (cycle.size() < 3)
        {
            outChains.push_back({labels, std::move(cycle), false});
            continue;
        }

        const std::size_t splitIndex = halfPerimeterVertex(cycle, positions);
        const Eigen::Vector3d startTangent = cyclicForwardTangent(cycle, 0, positions);
        const Eigen::Vector3d splitTangent = cyclicForwardTangent(
            cycle,
            splitIndex,
            positions
        );
        std::vector<uint32_t> firstArc(cycle.begin(), cycle.begin() + splitIndex + 1);
        std::vector<uint32_t> secondArc(cycle.begin() + splitIndex, cycle.end());
        secondArc.push_back(cycle.front());
        outChains.push_back({
            labels,
            std::move(firstArc),
            true,
            true,
            startTangent,
            splitTangent,
        });
        outChains.push_back({
            labels,
            std::move(secondArc),
            true,
            true,
            splitTangent,
            startTangent,
        });
    }
}

}

bool traceSegmentBorders(
    const CpuMesh& mesh,
    std::span<const uint32_t> triangleLabels,
    std::vector<TracedChain>& outChains,
    uint32_t& outBoundaryEdgeCount,
    std::string& error
)
{
    outChains.clear();
    outBoundaryEdgeCount = 0;
    error.clear();

    const std::size_t faceCount = mesh.faceAdjacency.size();
    std::map<LabelPair, std::set<TopologyEdge> > groupedEdges;
    for (std::size_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
    {
        const FaceAdjacency& adjacency = mesh.faceAdjacency[faceIndex];
        for (std::size_t edgeSlot = 0; edgeSlot < adjacency.neighbors.size(); ++edgeSlot)
        {
            const int32_t signedNeighbor = adjacency.neighbors[edgeSlot];
            if (signedNeighbor < 0)
            {
                continue;
            }
            const std::size_t neighborIndex = static_cast<std::size_t>(signedNeighbor);
            if (neighborIndex >= faceCount)
            {
                error = "Mesh adjacency contains an out-of-range neighboring face.";
                return false;
            }
            if (neighborIndex == faceIndex)
            {
                error = "Mesh adjacency contains a self-neighboring face.";
                return false;
            }
            if (triangleLabels[faceIndex] == triangleLabels[neighborIndex])
            {
                continue;
            }

            TopologyEdge edge;
            if (!topologyEdgeForFaceSlot(mesh, faceIndex, edgeSlot, edge, error))
            {
                return false;
            }
            if (!faceContainsEdge(mesh.faceTopologyVertexIndices[neighborIndex], edge))
            {
                error = "Adjacent faces disagree about their shared canonical edge.";
                return false;
            }

            const LabelPair labels = {
                std::min(triangleLabels[faceIndex], triangleLabels[neighborIndex]),
                std::max(triangleLabels[faceIndex], triangleLabels[neighborIndex]),
            };
            groupedEdges[labels].insert(edge);
        }
    }

    std::set<TopologyEdge> allBoundaryEdges;
    for (const auto& [labels, edges] : groupedEdges)
    {
        (void)labels;
        allBoundaryEdges.insert(edges.begin(), edges.end());
    }

    std::map<uint32_t, uint32_t> globalIncidentCounts;
    for (const TopologyEdge& edge : allBoundaryEdges)
    {
        if (!isFinite(mesh.canonicalPositions[edge.first])
            || !isFinite(mesh.canonicalPositions[edge.second]))
        {
            error = "A segment-border edge contains a non-finite canonical position.";
            return false;
        }
        ++globalIncidentCounts[edge.first];
        ++globalIncidentCounts[edge.second];
    }
    std::set<uint32_t> forcedSplitVertices;
    for (const auto& [vertex, incidentCount] : globalIncidentCounts)
    {
        if (incidentCount != 2)
        {
            forcedSplitVertices.insert(vertex);
        }
    }

    uint64_t boundaryEdgeCount = 0;
    std::vector<TracedChain> tracedChains;
    for (const auto& [labels, edges] : groupedEdges)
    {
        boundaryEdgeCount += edges.size();
        traceLabelPairEdges(
            labels,
            edges,
            mesh.canonicalPositions,
            forcedSplitVertices,
            tracedChains
        );
    }
    if (boundaryEdgeCount > std::numeric_limits<uint32_t>::max())
    {
        error = "The segment border contains too many edges for 32-bit overlay statistics.";
        return false;
    }
    outBoundaryEdgeCount = static_cast<uint32_t>(boundaryEdgeCount);

    outChains = std::move(tracedChains);
    return true;
}
}
