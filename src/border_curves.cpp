#include "border_curves.h"

#include <Eigen/Dense>

#include <algorithm>
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

namespace
{
constexpr double kCoincidentPointEpsilon = 1.0e-12;
constexpr uint32_t kMaximumDrawingSubdivisionDepth = 12;
constexpr uint32_t kMinimumFitSamplingDepth = 5;

struct LabelPair
{
    uint32_t first = 0;
    uint32_t second = 0;

    auto operator<=>(const LabelPair&) const = default;
};

struct TopologyEdge
{
    uint32_t first = 0;
    uint32_t second = 0;

    auto operator<=>(const TopologyEdge&) const = default;
};

struct TracedChain
{
    LabelPair labels;
    std::vector<uint32_t> vertexIndices;
    bool closedLoopArc = false;
    bool hasEndpointTangents = false;
    Eigen::Vector3d startTangent{};
    Eigen::Vector3d endTangent{};
};

struct CubicPiece
{
    std::array<segmesh::Float3, 4> controlPoints{};
    std::size_t firstSample = 0;
    std::size_t lastSample = 0;
    segmesh::BorderCurveFitStats fitStats{};
};

Eigen::Vector3d toEigen(const segmesh::Float3& point)
{
    return {
        static_cast<double>(point.x),
        static_cast<double>(point.y),
        static_cast<double>(point.z),
    };
}

segmesh::Float3 toFloat3(const Eigen::Vector3d& point)
{
    return {
        static_cast<float>(point.x()),
        static_cast<float>(point.y()),
        static_cast<float>(point.z()),
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

std::vector<segmesh::Float3> removeConsecutiveCoincidentPoints(
    std::span<const segmesh::Float3> points
)
{
    std::vector<segmesh::Float3> result;
    result.reserve(points.size());
    for (const segmesh::Float3& point : points)
    {
        if (result.empty() || pointDistance(result.back(), point) > kCoincidentPointEpsilon)
        {
            result.push_back(point);
        }
    }
    return result;
}

std::vector<double> chordLengthParameters(std::span<const segmesh::Float3> points)
{
    std::vector<double> parameters(points.size(), 0.0);
    for (std::size_t index = 1; index < points.size(); ++index)
    {
        parameters[index] = parameters[index - 1] + pointDistance(points[index - 1], points[index]);
    }

    const double totalLength = parameters.back();
    if (totalLength > kCoincidentPointEpsilon)
    {
        for (double& parameter : parameters)
        {
            parameter /= totalLength;
        }
    }
    return parameters;
}

std::array<segmesh::Float3, 4> lineEquivalentCubic(
    const segmesh::Float3& first,
    const segmesh::Float3& last
)
{
    const Eigen::Vector3d start = toEigen(first);
    const Eigen::Vector3d end = toEigen(last);
    const Eigen::Vector3d difference = end - start;
    return {
        first,
        toFloat3(start + difference / 3.0),
        toFloat3(start + difference * (2.0 / 3.0)),
        last,
    };
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

Eigen::Vector3d forwardTangentAt(
    std::span<const segmesh::Float3> points,
    std::size_t index
)
{
    Eigen::Vector3d tangent;
    if (index == 0)
    {
        tangent = toEigen(points[1]) - toEigen(points[0]);
    }
    else if (index + 1 == points.size())
    {
        tangent = toEigen(points[index]) - toEigen(points[index - 1]);
    }
    else
    {
        const Eigen::Vector3d incoming = (
            toEigen(points[index]) - toEigen(points[index - 1])
        ).normalized();
        const Eigen::Vector3d outgoing = (
            toEigen(points[index + 1]) - toEigen(points[index])
        ).normalized();
        tangent = incoming + outgoing;
        if (tangent.squaredNorm() <= kCoincidentPointEpsilon * kCoincidentPointEpsilon)
        {
            tangent = outgoing;
        }
    }
    return tangent.normalized();
}

double pointToLineDistance(
    const segmesh::Float3& point,
    const segmesh::Float3& lineStart,
    const segmesh::Float3& lineEnd
)
{
    const Eigen::Vector3d start = toEigen(lineStart);
    const Eigen::Vector3d direction = toEigen(lineEnd) - start;
    const double squaredLength = direction.squaredNorm();
    if (squaredLength <= kCoincidentPointEpsilon * kCoincidentPointEpsilon)
    {
        return (toEigen(point) - start).norm();
    }
    return ((toEigen(point) - start).cross(direction)).norm() / std::sqrt(squaredLength);
}

double pointToSegmentDistance(
    const segmesh::Float3& point,
    const segmesh::Float3& segmentStart,
    const segmesh::Float3& segmentEnd
)
{
    const Eigen::Vector3d start = toEigen(segmentStart);
    const Eigen::Vector3d direction = toEigen(segmentEnd) - start;
    const double squaredLength = direction.squaredNorm();
    if (squaredLength <= kCoincidentPointEpsilon * kCoincidentPointEpsilon)
    {
        return (toEigen(point) - start).norm();
    }
    const double parameter = std::clamp(
        (toEigen(point) - start).dot(direction) / squaredLength,
        0.0,
        1.0
    );
    return (toEigen(point) - (start + parameter * direction)).norm();
}

double cubicFlatness(const std::array<segmesh::Float3, 4>& cubic)
{
    const double perpendicularDeviation = std::max(
        pointToLineDistance(cubic[1], cubic[0], cubic[3]),
        pointToLineDistance(cubic[2], cubic[0], cubic[3])
    );
    const double controlPolygonLength = pointDistance(cubic[0], cubic[1])
        + pointDistance(cubic[1], cubic[2])
        + pointDistance(cubic[2], cubic[3]);
    const double chordLength = pointDistance(cubic[0], cubic[3]);
    return std::max(perpendicularDeviation, controlPolygonLength - chordLength);
}

void splitCubicAtHalf(
    const std::array<segmesh::Float3, 4>& cubic,
    std::array<segmesh::Float3, 4>& left,
    std::array<segmesh::Float3, 4>& right
)
{
    const Eigen::Vector3d p0 = toEigen(cubic[0]);
    const Eigen::Vector3d p1 = toEigen(cubic[1]);
    const Eigen::Vector3d p2 = toEigen(cubic[2]);
    const Eigen::Vector3d p3 = toEigen(cubic[3]);
    const Eigen::Vector3d p01 = (p0 + p1) * 0.5;
    const Eigen::Vector3d p12 = (p1 + p2) * 0.5;
    const Eigen::Vector3d p23 = (p2 + p3) * 0.5;
    const Eigen::Vector3d p012 = (p01 + p12) * 0.5;
    const Eigen::Vector3d p123 = (p12 + p23) * 0.5;
    const Eigen::Vector3d midpoint = (p012 + p123) * 0.5;
    left = {cubic[0], toFloat3(p01), toFloat3(p012), toFloat3(midpoint)};
    right = {toFloat3(midpoint), toFloat3(p123), toFloat3(p23), cubic[3]};
}

void sampleCubicForFit(
    const std::array<segmesh::Float3, 4>& cubic,
    double samplingTolerance,
    uint32_t depth,
    std::vector<segmesh::Float3>& outPoints
)
{
    if ((depth >= kMinimumFitSamplingDepth && cubicFlatness(cubic) <= samplingTolerance)
        || depth == kMaximumDrawingSubdivisionDepth)
    {
        outPoints.push_back(cubic.back());
        return;
    }

    std::array<segmesh::Float3, 4> left{};
    std::array<segmesh::Float3, 4> right{};
    splitCubicAtHalf(cubic, left, right);
    sampleCubicForFit(left, samplingTolerance, depth + 1, outPoints);
    sampleCubicForFit(right, samplingTolerance, depth + 1, outPoints);
}

double curveToPolylineDeviation(
    const std::array<segmesh::Float3, 4>& cubic,
    std::span<const segmesh::Float3> polyline,
    double fitTolerance,
    std::size_t& outSplitPoint
)
{
    std::vector<segmesh::Float3> curveSamples;
    curveSamples.reserve(65);
    curveSamples.push_back(cubic.front());
    const double chordLength = pointDistance(cubic.front(), cubic.back());
    const double samplingTolerance = std::max(
        std::min(fitTolerance * 0.125, chordLength * 1.0e-3),
        kCoincidentPointEpsilon
    );
    sampleCubicForFit(cubic, samplingTolerance, 0, curveSamples);

    double maximumDistance = 0.0;
    std::size_t nearestSegmentAtMaximum = 0;
    for (const segmesh::Float3& curvePoint : curveSamples)
    {
        double nearestDistance = std::numeric_limits<double>::infinity();
        std::size_t nearestSegment = 0;
        for (std::size_t segment = 0; segment + 1 < polyline.size(); ++segment)
        {
            const double distance = pointToSegmentDistance(
                curvePoint,
                polyline[segment],
                polyline[segment + 1]
            );
            if (distance < nearestDistance)
            {
                nearestDistance = distance;
                nearestSegment = segment;
            }
        }
        if (nearestDistance > maximumDistance)
        {
            maximumDistance = nearestDistance;
            nearestSegmentAtMaximum = nearestSegment;
        }
    }

    if (polyline.size() > 2)
    {
        outSplitPoint = std::clamp(
            nearestSegmentAtMaximum + 1,
            std::size_t{1},
            polyline.size() - 2
        );
    }
    else
    {
        outSplitPoint = 0;
    }
    return maximumDistance;
}

std::array<segmesh::Float3, 4> tangentCubicForSegment(
    const segmesh::Float3& first,
    const segmesh::Float3& last,
    Eigen::Vector3d startTangent,
    Eigen::Vector3d endTangent,
    double handleLength
)
{
    const Eigen::Vector3d start = toEigen(first);
    const Eigen::Vector3d end = toEigen(last);
    const Eigen::Vector3d chord = end - start;
    const Eigen::Vector3d chordDirection = chord.normalized();
    if (!startTangent.allFinite() || startTangent.dot(chordDirection) <= 0.0)
    {
        startTangent = chordDirection;
    }
    if (!endTangent.allFinite() || endTangent.dot(chordDirection) <= 0.0)
    {
        endTangent = chordDirection;
    }
    return {
        first,
        toFloat3(start + handleLength * startTangent.normalized()),
        toFloat3(end - handleLength * endTangent.normalized()),
        last,
    };
}

bool fitCubicWithTangents(
    std::span<const segmesh::Float3> points,
    const Eigen::Vector3d& startTangent,
    const Eigen::Vector3d& endTangent,
    std::array<segmesh::Float3, 4>& outControlPoints,
    segmesh::BorderCurveFitStats& outStats,
    std::size_t& outWorstPoint,
    std::string& error
)
{
    outControlPoints = {};
    outStats = {};
    outWorstPoint = 0;

    if (points.size() < 2)
    {
        error = "Bezier fitting requires at least two sample points.";
        return false;
    }
    if (!std::all_of(points.begin(), points.end(), isFinite)
        || !startTangent.allFinite() || !endTangent.allFinite())
    {
        error = "Bezier fitting received non-finite input.";
        return false;
    }
    if (points.size() == 2)
    {
        const double chordLength = pointDistance(points.front(), points.back());
        outControlPoints = tangentCubicForSegment(
            points.front(), points.back(), startTangent, endTangent, chordLength / 3.0
        );
        return true;
    }

    const std::vector<double> parameters = chordLengthParameters(points);
    const Eigen::Vector3d first = toEigen(points.front());
    const Eigen::Vector3d last = toEigen(points.back());
    const double chordLength = pointDistance(points.front(), points.back());

    const Eigen::Index rowCount = static_cast<Eigen::Index>((points.size() - 2) * 3);
    Eigen::MatrixXd design(rowCount, 2);
    Eigen::VectorXd rightHandSide(rowCount);
    for (std::size_t index = 1; index + 1 < points.size(); ++index)
    {
        const double t = parameters[index];
        const double oneMinusT = 1.0 - t;
        const double b0 = oneMinusT * oneMinusT * oneMinusT;
        const double b1 = 3.0 * oneMinusT * oneMinusT * t;
        const double b2 = 3.0 * oneMinusT * t * t;
        const double b3 = t * t * t;
        const Eigen::Index row = static_cast<Eigen::Index>((index - 1) * 3);
        design.block<3, 1>(row, 0) = b1 * startTangent;
        design.block<3, 1>(row, 1) = -b2 * endTangent;
        rightHandSide.segment<3>(row) = toEigen(points[index])
            - (b0 + b1) * first - (b2 + b3) * last;
    }

    const double fallbackHandle = chordLength / 3.0;
    double startHandle = fallbackHandle;
    double endHandle = fallbackHandle;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(design);
    const Eigen::VectorXd solution = solver.solve(rightHandSide);
    if (solver.rank() == 2 && solution.allFinite())
    {
        const double minimumHandle = std::max(chordLength * 1.0e-6, kCoincidentPointEpsilon);
        const double maximumHandle = std::max(chordLength, minimumHandle);
        if (solution[0] >= minimumHandle && solution[0] <= maximumHandle
            && solution[1] >= minimumHandle && solution[1] <= maximumHandle)
        {
            startHandle = solution[0];
            endHandle = solution[1];
        }
    }

    outControlPoints = {
        points.front(),
        toFloat3(first + startHandle * startTangent),
        toFloat3(last - endHandle * endTangent),
        points.back(),
    };
    if (!std::all_of(outControlPoints.begin(), outControlPoints.end(), isFinite))
    {
        outControlPoints = {};
        error = "Bezier fitting produced non-finite control points.";
        return false;
    }

    const std::span<const segmesh::Float3> cubic(outControlPoints);
    double squaredErrorSum = 0.0;
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const double fitError = pointDistance(
            points[index],
            segmesh::evaluateBezier(cubic, parameters[index])
        );
        if (!std::isfinite(fitError))
        {
            outControlPoints = {};
            outStats = {};
            error = "Bezier fitting produced a non-finite residual.";
            return false;
        }
        if (fitError > outStats.maxError)
        {
            outStats.maxError = fitError;
            outWorstPoint = index;
        }
        squaredErrorSum += fitError * fitError;
    }
    outStats.rmsError = std::sqrt(squaredErrorSum / static_cast<double>(points.size()));
    return true;
}

bool fitAdaptiveCubicPieces(
    std::span<const segmesh::Float3> points,
    double fitTolerance,
    const Eigen::Vector3d& chainStartTangent,
    const Eigen::Vector3d& chainEndTangent,
    std::vector<CubicPiece>& outPieces,
    std::string& error
)
{
    struct PendingPiece
    {
        std::size_t first = 0;
        std::size_t last = 0;
        Eigen::Vector3d startTangent{};
        Eigen::Vector3d endTangent{};
    };

    std::vector<PendingPiece> pending;
    pending.push_back({
        0,
        points.size() - 1,
        chainStartTangent,
        chainEndTangent,
    });

    while (!pending.empty())
    {
        PendingPiece current = pending.back();
        pending.pop_back();
        const std::span<const segmesh::Float3> piecePoints = points.subspan(
            current.first,
            current.last - current.first + 1
        );

        CubicPiece fitted;
        fitted.firstSample = current.first;
        fitted.lastSample = current.last;
        std::size_t worstLocalPoint = 0;
        if (!fitCubicWithTangents(
                piecePoints,
                current.startTangent,
                current.endTangent,
                fitted.controlPoints,
                fitted.fitStats,
                worstLocalPoint,
                error))
        {
            return false;
        }

        std::size_t geometricSplitPoint = 0;
        double geometricDeviation = curveToPolylineDeviation(
            fitted.controlPoints,
            piecePoints,
            fitTolerance,
            geometricSplitPoint
        );

        if (piecePoints.size() == 2)
        {
            const double chordLength = pointDistance(piecePoints.front(), piecePoints.back());
            double handleLength = chordLength / 3.0;
            for (uint32_t attempt = 0;
                 geometricDeviation > fitTolerance && attempt < 24;
                 ++attempt)
            {
                handleLength *= 0.5;
                fitted.controlPoints = tangentCubicForSegment(
                    piecePoints.front(),
                    piecePoints.back(),
                    current.startTangent,
                    current.endTangent,
                    handleLength
                );
                geometricDeviation = curveToPolylineDeviation(
                    fitted.controlPoints,
                    piecePoints,
                    fitTolerance,
                    geometricSplitPoint
                );
            }
            if (geometricDeviation > fitTolerance)
            {
                fitted.controlPoints = lineEquivalentCubic(
                    piecePoints.front(), piecePoints.back()
                );
                geometricDeviation = 0.0;
            }
            fitted.fitStats.maxError = std::max(
                fitted.fitStats.maxError,
                geometricDeviation
            );
            outPieces.push_back(std::move(fitted));
            continue;
        }

        if (geometricDeviation > fitted.fitStats.maxError)
        {
            fitted.fitStats.maxError = geometricDeviation;
            worstLocalPoint = geometricSplitPoint;
        }

        if (fitted.fitStats.maxError <= fitTolerance)
        {
            outPieces.push_back(std::move(fitted));
            continue;
        }

        if (worstLocalPoint == 0 || worstLocalPoint + 1 == piecePoints.size())
        {
            worstLocalPoint = piecePoints.size() / 2;
        }
        const std::size_t split = current.first + worstLocalPoint;
        const Eigen::Vector3d splitTangent = forwardTangentAt(points, split);

        // Push right first so the LIFO traversal emits pieces in chain order.
        pending.push_back({split, current.last, splitTangent, current.endTangent});
        pending.push_back({current.first, split, current.startTangent, splitTangent});
    }
    return true;
}

void flattenCubicRecursive(
    const std::array<segmesh::Float3, 4>& cubic,
    double tolerance,
    uint32_t depth,
    std::vector<segmesh::Float3>& outPoints,
    bool& outSubdivisionCapped
)
{
    const double flatness = cubicFlatness(cubic);
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
    splitCubicAtHalf(cubic, left, right);
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
Float3 evaluateBezier(std::span<const Float3> controlPoints, double t)
{
    if (controlPoints.empty())
    {
        return {};
    }
    if (controlPoints.size() == 1 || t <= 0.0)
    {
        return controlPoints.front();
    }
    if (t >= 1.0)
    {
        return controlPoints.back();
    }

    std::vector<Eigen::Vector3d> work;
    work.reserve(controlPoints.size());
    for (const Float3& point : controlPoints)
    {
        work.push_back(toEigen(point));
    }

    for (std::size_t remaining = work.size() - 1; remaining > 0; --remaining)
    {
        for (std::size_t index = 0; index < remaining; ++index)
        {
            work[index] = (1.0 - t) * work[index] + t * work[index + 1];
        }
    }
    return toFloat3(work.front());
}

bool fitEndpointConstrainedCubic(
    std::span<const Float3> points,
    std::array<Float3, 4>& outControlPoints,
    BorderCurveFitStats& outStats,
    std::string& error
)
{
    error.clear();
    outControlPoints = {};
    outStats = {};

    for (const Float3& point : points)
    {
        if (!isFinite(point))
        {
            error = "Bezier fitting received a non-finite sample point.";
            return false;
        }
    }

    const std::vector<Float3> samples = removeConsecutiveCoincidentPoints(points);
    if (samples.size() < 2)
    {
        error = "Bezier fitting requires at least two distinct consecutive sample points.";
        return false;
    }

    std::size_t worstPoint = 0;
    return fitCubicWithTangents(
        samples,
        forwardTangentAt(samples, 0),
        forwardTangentAt(samples, samples.size() - 1),
        outControlPoints,
        outStats,
        worstPoint,
        error
    );
}

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
    outOverlay.boundaryEdgeCount = static_cast<uint32_t>(boundaryEdgeCount);

    double totalSquaredFitError = 0.0;
    uint64_t totalFittedPointCount = 0;
    for (const TracedChain& chain : tracedChains)
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

        std::vector<Float3> fittedPoints = removeConsecutiveCoincidentPoints(sourcePoints);
        if (fittedPoints.size() < 2)
        {
            ++outOverlay.skippedChainCount;
            continue;
        }

        Eigen::Vector3d chainStartTangent = forwardTangentAt(fittedPoints, 0);
        Eigen::Vector3d chainEndTangent = forwardTangentAt(
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

        std::vector<CubicPiece> pieces;
        if (!fitAdaptiveCubicPieces(
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
        for (const CubicPiece& piece : pieces)
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
