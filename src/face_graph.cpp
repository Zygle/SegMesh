#include "face_graph.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace
{
constexpr double kCoplanarNormalDifferenceEpsilon = 1.0e-8;
constexpr double kCoplanarPlaneDistanceEpsilon = 1.0e-7;

double dot(const segmesh::Float3& a, const segmesh::Float3& b)
{
    return static_cast<double>(a.x) * static_cast<double>(b.x)
        + static_cast<double>(a.y) * static_cast<double>(b.y)
        + static_cast<double>(a.z) * static_cast<double>(b.z);
}

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

    const Eigen::Vector3d faceNormal = toVector3d(mesh.faceNormals[static_cast<std::size_t>(faceIndex)]);
    const Eigen::Vector3d neighborNormal = toVector3d(mesh.faceNormals[static_cast<std::size_t>(neighborIndex)]);
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
double faceNormalDot(const CpuMesh& mesh, uint32_t faceIndex, uint32_t neighborIndex)
{
    return std::clamp(
        dot(
            mesh.faceNormals[static_cast<std::size_t>(faceIndex)],
            mesh.faceNormals[static_cast<std::size_t>(neighborIndex)]
        ),
        -1.0,
        1.0
    );
}

double edgeDifferenceForMode(
    const segmesh::CpuMesh& mesh,
    segmesh::SegmentationModelType segmentationModelType,
    uint32_t faceIndex,
    std::size_t edgeSlot,
    uint32_t neighborIndex
)
{
    const double normalDot = faceNormalDot(mesh, faceIndex, neighborIndex);

    if (segmentationModelType == segmesh::SegmentationModelType::Graphical)
    {
        const double rawDifference =
            static_cast<double>(mesh.faceAdjacency[static_cast<std::size_t>(faceIndex)].concavityScales[edgeSlot])
            * (1.0 - normalDot);
        return rawDifference / std::max(static_cast<double>(mesh.averageGraphicalDifference), 1.0e-12);
    }

    const double rawNormalDifference = 1.0 - normalDot;
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
    if (mesh.faceNormals.size() != faceCount || mesh.faceAdjacency.size() != faceCount)
    {
        error = "Mesh topology data is incomplete. Reload the mesh before segmenting.";
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
            const double edgeLength = std::max(static_cast<double>(adjacency.edgeLengths[edgeSlot]), 1.0e-12);

            outEdges.push_back({faceIndex, static_cast<uint32_t>(neighborIndex), edgeLength, difference});
        }
    }

    if (outEdges.empty())
    {
        error = "The mesh face graph has no adjacency edges to solve over.";
        return false;
    }

    outAverageDifference = 1.0;
    return true;
}

}
