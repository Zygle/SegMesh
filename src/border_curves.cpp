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
constexpr uint32_t kHardMaximumDegree = 64;

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

double maximumControlPolygonDeviation(
    std::span<const segmesh::Float3> controlPoints,
    const std::array<segmesh::Float3, 4>& cubicControlPoints
)
{
    if (controlPoints.size() < 2)
    {
        return 0.0;
    }

    const std::span<const segmesh::Float3> cubic(cubicControlPoints);
    const double degree = static_cast<double>(controlPoints.size() - 1);
    double maximumDeviation = 0.0;
    for (std::size_t index = 0; index < controlPoints.size(); ++index)
    {
        const double parameter = static_cast<double>(index) / degree;
        maximumDeviation = std::max(
            maximumDeviation,
            pointDistance(controlPoints[index], segmesh::evaluateBezier(cubic, parameter))
        );
    }
    return maximumDeviation;
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
        std::vector<uint32_t> firstArc(cycle.begin(), cycle.begin() + splitIndex + 1);
        std::vector<uint32_t> secondArc(cycle.begin() + splitIndex, cycle.end());
        secondArc.push_back(cycle.front());
        outChains.push_back({labels, std::move(firstArc), true});
        outChains.push_back({labels, std::move(secondArc), true});
    }
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

std::vector<Float3> elevateBezierDegree(std::span<const Float3> controlPoints)
{
    if (controlPoints.empty())
    {
        return {};
    }

    const std::size_t oldDegree = controlPoints.size() - 1;
    const std::size_t newDegree = oldDegree + 1;
    std::vector<Float3> elevated(controlPoints.size() + 1);
    elevated.front() = controlPoints.front();
    elevated.back() = controlPoints.back();
    for (std::size_t index = 1; index <= oldDegree; ++index)
    {
        const double previousWeight = static_cast<double>(index) / static_cast<double>(newDegree);
        const Eigen::Vector3d point = previousWeight * toEigen(controlPoints[index - 1])
            + (1.0 - previousWeight) * toEigen(controlPoints[index]);
        elevated[index] = toFloat3(point);
    }
    return elevated;
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

    const std::vector<double> parameters = chordLengthParameters(samples);
    if (samples.size() == 2)
    {
        outControlPoints = lineEquivalentCubic(samples.front(), samples.back());
    }
    else if (samples.size() == 3)
    {
        const double t = parameters[1];
        const double oneMinusT = 1.0 - t;
        const double denominator = 2.0 * oneMinusT * t;
        if (denominator <= std::numeric_limits<double>::epsilon())
        {
            outControlPoints = lineEquivalentCubic(samples.front(), samples.back());
        }
        else
        {
            const Eigen::Vector3d quadraticMiddle = (
                toEigen(samples[1])
                - oneMinusT * oneMinusT * toEigen(samples.front())
                - t * t * toEigen(samples.back())
            ) / denominator;
            outControlPoints = {
                samples.front(),
                toFloat3(toEigen(samples.front()) / 3.0 + quadraticMiddle * (2.0 / 3.0)),
                toFloat3(quadraticMiddle * (2.0 / 3.0) + toEigen(samples.back()) / 3.0),
                samples.back(),
            };
        }
    }
    else
    {
        Eigen::MatrixXd design(static_cast<Eigen::Index>(samples.size() - 2), 2);
        Eigen::MatrixXd rightHandSide(static_cast<Eigen::Index>(samples.size() - 2), 3);
        const Eigen::Vector3d first = toEigen(samples.front());
        const Eigen::Vector3d last = toEigen(samples.back());

        for (std::size_t index = 1; index + 1 < samples.size(); ++index)
        {
            const double t = parameters[index];
            const double oneMinusT = 1.0 - t;
            const double b0 = oneMinusT * oneMinusT * oneMinusT;
            const double b1 = 3.0 * oneMinusT * oneMinusT * t;
            const double b2 = 3.0 * oneMinusT * t * t;
            const double b3 = t * t * t;
            const Eigen::Index row = static_cast<Eigen::Index>(index - 1);
            design(row, 0) = b1;
            design(row, 1) = b2;
            rightHandSide.row(row) = (toEigen(samples[index]) - b0 * first - b3 * last).transpose();
        }

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(design);
        const Eigen::MatrixXd solution = solver.solve(rightHandSide);
        if (solver.rank() < 2 || !solution.allFinite())
        {
            outControlPoints = lineEquivalentCubic(samples.front(), samples.back());
        }
        else
        {
            outControlPoints = {
                samples.front(),
                toFloat3(solution.row(0).transpose()),
                toFloat3(solution.row(1).transpose()),
                samples.back(),
            };
        }
    }

    if (!std::all_of(outControlPoints.begin(), outControlPoints.end(), isFinite))
    {
        outControlPoints = {};
        error = "Bezier fitting produced non-finite control points.";
        return false;
    }

    double squaredErrorSum = 0.0;
    const std::span<const Float3> cubic(outControlPoints);
    for (std::size_t index = 0; index < samples.size(); ++index)
    {
        const double fitError = pointDistance(samples[index], evaluateBezier(cubic, parameters[index]));
        if (!std::isfinite(fitError))
        {
            outControlPoints = {};
            outStats = {};
            error = "Bezier fitting produced a non-finite residual.";
            return false;
        }
        outStats.maxError = std::max(outStats.maxError, fitError);
        squaredErrorSum += fitError * fitError;
    }
    outStats.rmsError = std::sqrt(squaredErrorSum / static_cast<double>(samples.size()));
    return true;
}

std::vector<Float3> adaptiveDegreeElevate(
    const std::array<Float3, 4>& cubicControlPoints,
    const BorderCurveSettings& settings,
    bool& outDegreeCapped
)
{
    const double tolerance = std::isfinite(settings.tolerance) && settings.tolerance >= 0.0
        ? settings.tolerance
        : BorderCurveSettings{}.tolerance;
    const uint32_t maximumDegree = std::clamp(settings.maxDegree, 3u, kHardMaximumDegree);

    std::vector<Float3> controlPoints(cubicControlPoints.begin(), cubicControlPoints.end());
    double deviation = maximumControlPolygonDeviation(controlPoints, cubicControlPoints);
    while (deviation > tolerance && controlPoints.size() - 1 < maximumDegree)
    {
        controlPoints = elevateBezierDegree(controlPoints);
        deviation = maximumControlPolygonDeviation(controlPoints, cubicControlPoints);
    }

    outDegreeCapped = deviation > tolerance && controlPoints.size() - 1 >= maximumDegree;
    return controlPoints;
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
    if (!std::isfinite(settings.tolerance) || settings.tolerance < 0.0)
    {
        error = "Border-curve tolerance must be a finite non-negative value.";
        return false;
    }
    if (settings.maxDegree < 3 || settings.maxDegree > kHardMaximumDegree)
    {
        error = "Border-curve maximum degree must be between 3 and 64.";
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

        std::array<Float3, 4> cubic{};
        BorderCurveFitStats fitStats{};
        if (!fitEndpointConstrainedCubic(fittedPoints, cubic, fitStats, error))
        {
            return false;
        }

        bool degreeCapped = false;
        const std::vector<Float3> elevated = adaptiveDegreeElevate(cubic, settings, degreeCapped);
        if (!appendPolyline(elevated, outOverlay.vertices, outOverlay.indices, error)
            || !appendPolyline(cubic, outOverlay.controlVertices, outOverlay.controlIndices, error))
        {
            return false;
        }

        if (degreeCapped)
        {
            ++outOverlay.cappedChainCount;
        }
        outOverlay.maxFitError = std::max(outOverlay.maxFitError, fitStats.maxError);
        totalSquaredFitError += fitStats.rmsError * fitStats.rmsError
            * static_cast<double>(fittedPoints.size());
        totalFittedPointCount += fittedPoints.size();
        outOverlay.chains.push_back({
            chain.labels.first,
            chain.labels.second,
            static_cast<uint32_t>(sourcePoints.size()),
            static_cast<uint32_t>(fittedPoints.size()),
            static_cast<uint32_t>(elevated.size() - 1),
            fitStats.maxError,
            fitStats.rmsError,
            degreeCapped,
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
