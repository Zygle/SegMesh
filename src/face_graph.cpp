#include "face_graph.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr double kCoplanarNormalDifferenceEpsilon = 1.0e-8;
constexpr double kCoplanarPlaneDistanceEpsilon = 1.0e-7;

Eigen::Vector3d toVector3d(const segmesh::Float3& value)
{
    return Eigen::Vector3d(
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z)
    );
}

Eigen::Vector3d triangleVertexPosition(const segmesh::CpuMesh& mesh, uint32_t faceIndex, std::size_t corner)
{
    const std::size_t indexOffset = static_cast<std::size_t>(faceIndex) * 3 + corner;
    const uint32_t vertexIndex = mesh.indices[indexOffset];
    const segmesh::MeshVertex& vertex = mesh.vertices[static_cast<std::size_t>(vertexIndex)];
    return Eigen::Vector3d(
        static_cast<double>(vertex.x),
        static_cast<double>(vertex.y),
        static_cast<double>(vertex.z)
    );
}

double maxTriangleEdgeLength(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2
)
{
    return std::max({(p0 - p1).norm(), (p1 - p2).norm(), (p2 - p0).norm()});
}

bool areFacesGeometricallyCoplanar(
    const segmesh::CpuMesh& mesh,
    uint32_t faceIndex,
    uint32_t neighborIndex,
    double rawNormalDifference
)
{
    if (rawNormalDifference > kCoplanarNormalDifferenceEpsilon)
    {
        return false;
    }

    const std::size_t faceIndexOffset = static_cast<std::size_t>(faceIndex) * 3;
    const std::size_t neighborIndexOffset = static_cast<std::size_t>(neighborIndex) * 3;
    if (faceIndexOffset + 2 >= mesh.indices.size() || neighborIndexOffset + 2 >= mesh.indices.size())
    {
        return false;
    }

    const Eigen::Vector3d faceNormal = toVector3d(mesh.faceNormals[static_cast<std::size_t>(faceIndex)]).normalized();
    const Eigen::Vector3d neighborNormal = toVector3d(mesh.faceNormals[static_cast<std::size_t>(neighborIndex)]).normalized();
    const Eigen::Vector3d p0 = triangleVertexPosition(mesh, faceIndex, 0);
    const Eigen::Vector3d p1 = triangleVertexPosition(mesh, faceIndex, 1);
    const Eigen::Vector3d p2 = triangleVertexPosition(mesh, faceIndex, 2);
    const Eigen::Vector3d q0 = triangleVertexPosition(mesh, neighborIndex, 0);
    const Eigen::Vector3d q1 = triangleVertexPosition(mesh, neighborIndex, 1);
    const Eigen::Vector3d q2 = triangleVertexPosition(mesh, neighborIndex, 2);

    const double maxEdgeLength = std::max(
        maxTriangleEdgeLength(p0, p1, p2),
        maxTriangleEdgeLength(q0, q1, q2)
    );
    const double tolerance = kCoplanarPlaneDistanceEpsilon * std::max(maxEdgeLength, 1.0);

    const double p0ToNeighborPlane = std::abs((p0 - q0).dot(neighborNormal));
    const double p1ToNeighborPlane = std::abs((p1 - q0).dot(neighborNormal));
    const double p2ToNeighborPlane = std::abs((p2 - q0).dot(neighborNormal));
    const double q0ToFacePlane = std::abs((q0 - p0).dot(faceNormal));
    const double q1ToFacePlane = std::abs((q1 - p0).dot(faceNormal));
    const double q2ToFacePlane = std::abs((q2 - p0).dot(faceNormal));

    return p0ToNeighborPlane <= tolerance
        && p1ToNeighborPlane <= tolerance
        && p2ToNeighborPlane <= tolerance
        && q0ToFacePlane <= tolerance
        && q1ToFacePlane <= tolerance
        && q2ToFacePlane <= tolerance;
}

}

namespace segmesh::detail
{
double faceNormalDifference(const CpuMesh& mesh, uint32_t faceIndex, uint32_t neighborIndex)
{
    const Eigen::Vector3d first = toVector3d(mesh.faceNormals[faceIndex]);
    const Eigen::Vector3d second = toVector3d(mesh.faceNormals[neighborIndex]);
    if (!first.allFinite() || !second.allFinite()
        || first.squaredNorm() == 0.0 || second.squaredNorm() == 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Equivalent to 1 - dot for unit normals, without cancellation near coplanarity.
    return std::clamp(0.5 * (first.normalized() - second.normalized()).squaredNorm(), 0.0, 2.0);
}

double edgeDifferenceForMode(
    const segmesh::CpuMesh& mesh,
    segmesh::SegmentationModelType segmentationModelType,
    uint32_t faceIndex,
    std::size_t edgeSlot,
    uint32_t neighborIndex
)
{
    const double rawNormalDifference = faceNormalDifference(mesh, faceIndex, neighborIndex);

    if (segmentationModelType == segmesh::SegmentationModelType::Graphical)
    {
        const double rawDifference =
            static_cast<double>(mesh.faceAdjacency[static_cast<std::size_t>(faceIndex)].concavityScales[edgeSlot])
            * rawNormalDifference;
        return rawDifference / std::max(static_cast<double>(mesh.averageGraphicalDifference), 1.0e-12);
    }

    const double normalDifference = rawNormalDifference
        / std::max(static_cast<double>(mesh.averageEngineeringNormalDifference), 1.0e-12);
    if (areFacesGeometricallyCoplanar(mesh, faceIndex, neighborIndex, rawNormalDifference))
    {
        // Coplanar triangles should stay in the same engineering patch even if noisy curvature estimates differ.
        return normalDifference;
    }

    const double gaussianDifference = std::abs(
        static_cast<double>(mesh.faceGaussianCurvatures[static_cast<std::size_t>(faceIndex)])
            - static_cast<double>(mesh.faceGaussianCurvatures[static_cast<std::size_t>(neighborIndex)])
    ) / std::max(static_cast<double>(mesh.averageGaussianDifference), 1.0e-12);
    const double meanDifference = std::abs(
        static_cast<double>(mesh.faceMeanCurvatures[static_cast<std::size_t>(faceIndex)])
            - static_cast<double>(mesh.faceMeanCurvatures[static_cast<std::size_t>(neighborIndex)])
    ) / std::max(static_cast<double>(mesh.averageMeanDifference), 1.0e-12);

    return std::max({normalDifference, gaussianDifference, meanDifference});
}

bool validateFaceGraphData(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    std::string& error
)
{
    error.clear();
    const std::size_t faceCount = mesh.indices.size() / 3;
    if (mesh.indices.size() % 3 != 0
        || faceCount > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())
        || mesh.faceNormals.size() != faceCount || mesh.faceAdjacency.size() != faceCount)
    {
        error = "Mesh topology data is incomplete or exceeds the supported face count.";
        return false;
    }

    const bool engineering = segmentationModelType == SegmentationModelType::Engineering;
    if (engineering && (mesh.faceGaussianCurvatures.size() != faceCount
        || mesh.faceMeanCurvatures.size() != faceCount))
    {
        error = "Mesh curvature data is incomplete.";
        return false;
    }
    const auto validScale = [](float value) { return std::isfinite(value) && value > 0.0f; };
    if ((!engineering && !validScale(mesh.averageGraphicalDifference))
        || (engineering && (!validScale(mesh.averageEngineeringNormalDifference)
            || !validScale(mesh.averageGaussianDifference) || !validScale(mesh.averageMeanDifference))))
    {
        error = "Mesh difference normalization must be finite and positive.";
        return false;
    }

    for (const uint32_t vertex : mesh.indices)
    {
        if (vertex >= mesh.vertices.size())
        {
            error = "Mesh contains an out-of-range vertex index.";
            return false;
        }
    }
    for (std::size_t face = 0; face < faceCount; ++face)
    {
        const Eigen::Vector3d normal = toVector3d(mesh.faceNormals[face]);
        if (!normal.allFinite() || normal.squaredNorm() == 0.0)
        {
            error = "Mesh contains a non-finite or zero-length face normal.";
            return false;
        }
        if (engineering && (!std::isfinite(mesh.faceGaussianCurvatures[face])
            || !std::isfinite(mesh.faceMeanCurvatures[face])))
        {
            error = "Mesh contains non-finite curvature values.";
            return false;
        }
        const FaceAdjacency& adjacency = mesh.faceAdjacency[face];
        for (std::size_t slot = 0; slot < adjacency.neighbors.size(); ++slot)
        {
            const int32_t neighbor = adjacency.neighbors[slot];
            if (neighbor < 0)
            {
                continue;
            }
            if (static_cast<std::size_t>(neighbor) >= faceCount
                || static_cast<std::size_t>(neighbor) == face)
            {
                error = "Mesh adjacency contains an invalid neighboring face.";
                return false;
            }
            if (!std::isfinite(adjacency.edgeLengths[slot]) || adjacency.edgeLengths[slot] < 0.0f
                || !std::isfinite(adjacency.concavityScales[slot]) || adjacency.concavityScales[slot] < 0.0f)
            {
                error = "Mesh adjacency contains an invalid edge length or concavity scale.";
                return false;
            }
        }
    }
    return true;
}

bool buildFaceGraphEdges(
    const segmesh::CpuMesh& mesh,
    segmesh::SegmentationModelType segmentationModelType,
    std::vector<FaceGraphEdgeInfo>& outEdges,
    double& outAverageDifference,
    std::string& error
)
{
    outEdges.clear();
    outAverageDifference = 0.0;
    error.clear();

    const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (!validateFaceGraphData(mesh, segmentationModelType, error))
    {
        return false;
    }

    outEdges.reserve(static_cast<std::size_t>(faceCount) * 3 / 2);

    for (uint32_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
    {
        const segmesh::FaceAdjacency& adjacency = mesh.faceAdjacency[static_cast<std::size_t>(faceIndex)];
        for (std::size_t edgeSlot = 0; edgeSlot < adjacency.neighbors.size(); ++edgeSlot)
        {
            const int32_t neighborIndex = adjacency.neighbors[edgeSlot];
            if (neighborIndex < 0 || faceIndex >= static_cast<uint32_t>(neighborIndex))
            {
                continue;
            }

            const double difference =
                edgeDifferenceForMode(mesh, segmentationModelType, faceIndex, edgeSlot, static_cast<uint32_t>(neighborIndex));
            if (!std::isfinite(difference) || difference < 0.0)
            {
                error = "Mesh graph contains a non-finite or negative edge difference.";
                return false;
            }
            const double edgeLength = std::max(static_cast<double>(adjacency.edgeLengths[edgeSlot]), 1.0e-12);

            outEdges.push_back({faceIndex, static_cast<uint32_t>(neighborIndex), edgeLength, difference});
        }
    }

    // Disconnected isolated faces form a valid graph with no merge candidates.
    outAverageDifference = 1.0;
    return true;
}

}
