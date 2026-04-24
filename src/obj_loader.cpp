#include "obj_loader.h"

#include <Eigen/Dense>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{
using OpenMeshTriMesh = OpenMesh::TriMesh_ArrayKernelT<>;
using segmesh::FaceAdjacency;
using segmesh::Float3;
using segmesh::MeshVertex;

constexpr double kPi = 3.14159265358979323846;

Eigen::Vector3f toEigen(const OpenMeshTriMesh::Point& point)
{
    return {point[0], point[1], point[2]};
}

Float3 toFloat3(const Eigen::Vector3f& v)
{
    return {v.x(), v.y(), v.z()};
}

Eigen::Vector3f normalizeSafe(const Eigen::Vector3f& v, const Eigen::Vector3f& fallback = {0.0f, 1.0f, 0.0f})
{
    const float len = v.norm();
    if (len < 1.0e-12f)
    {
        return fallback;
    }

    return v / len;
}

double safeAngleBetween(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
    const double lenProduct = a.norm() * b.norm();
    if (lenProduct < 1.0e-16)
    {
        return 0.0;
    }

    const double cosine = std::clamp(a.dot(b) / lenProduct, -1.0, 1.0);
    return std::acos(cosine);
}

double cotangentAtVertex(
    const Eigen::Vector3d& vertex,
    const Eigen::Vector3d& edgePoint0,
    const Eigen::Vector3d& edgePoint1
)
{
    const Eigen::Vector3d e0 = edgePoint0 - vertex;
    const Eigen::Vector3d e1 = edgePoint1 - vertex;
    const double crossNorm = e0.cross(e1).norm();
    if (crossNorm < 1.0e-16)
    {
        return 0.0;
    }

    return e0.dot(e1) / crossNorm;
}

std::vector<double> smoothVertexScalarField(
    const OpenMeshTriMesh& mesh,
    const std::vector<double>& values,
    int iterations
)
{
    std::vector<double> smoothed = values;
    std::vector<double> scratch(values.size(), 0.0);

    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        for (const OpenMeshTriMesh::VertexHandle vh : mesh.vertices())
        {
            double neighborSum = 0.0;
            uint32_t neighborCount = 0;
            for (auto vvIt = mesh.cvv_iter(vh); vvIt.is_valid(); ++vvIt)
            {
                neighborSum += smoothed[static_cast<std::size_t>(vvIt->idx())];
                ++neighborCount;
            }

            const double currentValue = smoothed[static_cast<std::size_t>(vh.idx())];
            if (neighborCount == 0)
            {
                scratch[static_cast<std::size_t>(vh.idx())] = currentValue;
                continue;
            }

            const double neighborAverage = neighborSum / static_cast<double>(neighborCount);
            scratch[static_cast<std::size_t>(vh.idx())] = 0.5 * currentValue + 0.5 * neighborAverage;
        }

        smoothed.swap(scratch);
    }

    return smoothed;
}

void computeFaceCurvatures(
    const OpenMeshTriMesh& mesh,
    const std::vector<Eigen::Vector3f>& preprocessedPositions,
    const std::vector<int32_t>& faceMap,
    std::vector<float>& outFaceGaussianCurvatures,
    std::vector<float>& outFaceMeanCurvatures
)
{
    const std::size_t vertexCount = mesh.n_vertices();
    std::vector<double> vertexAreas(vertexCount, 0.0);
    std::vector<double> vertexAngleSums(vertexCount, 0.0);
    std::vector<Eigen::Vector3d> laplaceVectors(vertexCount, Eigen::Vector3d::Zero());

    for (const OpenMeshTriMesh::FaceHandle fh : mesh.faces())
    {
        std::array<OpenMeshTriMesh::VertexHandle, 3> faceVertices = {
            OpenMeshTriMesh::VertexHandle(),
            OpenMeshTriMesh::VertexHandle(),
            OpenMeshTriMesh::VertexHandle(),
        };
        std::size_t corner = 0;
        for (auto fvIt = mesh.cfv_iter(fh); fvIt.is_valid(); ++fvIt)
        {
            if (corner < faceVertices.size())
            {
                faceVertices[corner] = *fvIt;
            }
            ++corner;
        }

        if (corner != 3)
        {
            continue;
        }

        const Eigen::Vector3d p0 = preprocessedPositions[faceVertices[0].idx()].cast<double>();
        const Eigen::Vector3d p1 = preprocessedPositions[faceVertices[1].idx()].cast<double>();
        const Eigen::Vector3d p2 = preprocessedPositions[faceVertices[2].idx()].cast<double>();
        const double area = 0.5 * (p1 - p0).cross(p2 - p0).norm();
        const double areaContribution = area / 3.0;

        const std::array<double, 3> angles = {
            safeAngleBetween(p1 - p0, p2 - p0),
            safeAngleBetween(p2 - p1, p0 - p1),
            safeAngleBetween(p0 - p2, p1 - p2),
        };

        for (std::size_t i = 0; i < 3; ++i)
        {
            const std::size_t vertexIndex = static_cast<std::size_t>(faceVertices[i].idx());
            vertexAreas[vertexIndex] += areaContribution;
            vertexAngleSums[vertexIndex] += angles[i];
        }
    }

    for (const OpenMeshTriMesh::EdgeHandle eh : mesh.edges())
    {
        const OpenMeshTriMesh::HalfedgeHandle heh0 = mesh.halfedge_handle(eh, 0);
        const OpenMeshTriMesh::HalfedgeHandle heh1 = mesh.halfedge_handle(eh, 1);
        const OpenMeshTriMesh::VertexHandle fromVh = mesh.from_vertex_handle(heh0);
        const OpenMeshTriMesh::VertexHandle toVh = mesh.to_vertex_handle(heh0);

        const Eigen::Vector3d pFrom = preprocessedPositions[fromVh.idx()].cast<double>();
        const Eigen::Vector3d pTo = preprocessedPositions[toVh.idx()].cast<double>();

        double cotangentSum = 0.0;
        for (const OpenMeshTriMesh::HalfedgeHandle heh : {heh0, heh1})
        {
            if (mesh.is_boundary(heh))
            {
                continue;
            }

            const OpenMeshTriMesh::HalfedgeHandle nextHeh = mesh.next_halfedge_handle(heh);
            const OpenMeshTriMesh::VertexHandle oppositeVh = mesh.to_vertex_handle(nextHeh);
            const Eigen::Vector3d pOpposite = preprocessedPositions[oppositeVh.idx()].cast<double>();
            cotangentSum += cotangentAtVertex(pOpposite, pFrom, pTo);
        }

        const Eigen::Vector3d edgeVector = pFrom - pTo;
        laplaceVectors[static_cast<std::size_t>(fromVh.idx())] += cotangentSum * edgeVector;
        laplaceVectors[static_cast<std::size_t>(toVh.idx())] -= cotangentSum * edgeVector;
    }

    std::vector<double> gaussianCurvatures(vertexCount, 0.0);
    std::vector<double> meanCurvatures(vertexCount, 0.0);
    for (const OpenMeshTriMesh::VertexHandle vh : mesh.vertices())
    {
        const std::size_t vertexIndex = static_cast<std::size_t>(vh.idx());
        const double area = std::max(vertexAreas[vertexIndex], 1.0e-12);
        const double targetAngle = mesh.is_boundary(vh) ? kPi : 2.0 * kPi;
        gaussianCurvatures[vertexIndex] = (targetAngle - vertexAngleSums[vertexIndex]) / area;
        meanCurvatures[vertexIndex] = laplaceVectors[vertexIndex].norm() / (4.0 * area);
    }

    gaussianCurvatures = smoothVertexScalarField(mesh, gaussianCurvatures, 2);
    meanCurvatures = smoothVertexScalarField(mesh, meanCurvatures, 2);

    outFaceGaussianCurvatures.assign(outFaceGaussianCurvatures.size(), 0.0f);
    outFaceMeanCurvatures.assign(outFaceMeanCurvatures.size(), 0.0f);
    for (const OpenMeshTriMesh::FaceHandle fh : mesh.faces())
    {
        const int32_t outputFaceIndex = faceMap[fh.idx()];
        if (outputFaceIndex < 0)
        {
            continue;
        }

        std::array<OpenMeshTriMesh::VertexHandle, 3> faceVertices = {
            OpenMeshTriMesh::VertexHandle(),
            OpenMeshTriMesh::VertexHandle(),
            OpenMeshTriMesh::VertexHandle(),
        };
        std::size_t corner = 0;
        for (auto fvIt = mesh.cfv_iter(fh); fvIt.is_valid(); ++fvIt)
        {
            if (corner < faceVertices.size())
            {
                faceVertices[corner] = *fvIt;
            }
            ++corner;
        }

        if (corner != 3)
        {
            continue;
        }

        double gaussian = 0.0;
        double mean = 0.0;
        for (const OpenMeshTriMesh::VertexHandle vh : faceVertices)
        {
            gaussian += gaussianCurvatures[static_cast<std::size_t>(vh.idx())];
            mean += meanCurvatures[static_cast<std::size_t>(vh.idx())];
        }

        outFaceGaussianCurvatures[static_cast<std::size_t>(outputFaceIndex)] = static_cast<float>(gaussian / 3.0);
        outFaceMeanCurvatures[static_cast<std::size_t>(outputFaceIndex)] = static_cast<float>(mean / 3.0);
    }
}

uint32_t packNormal(const Eigen::Vector3f& normal)
{
    auto toByte = [](float value) -> uint8_t
    {
        const float normalized = std::clamp(value * 0.5f + 0.5f, 0.0f, 1.0f);
        return static_cast<uint8_t>(normalized * 255.0f + 0.5f);
    };

    const uint32_t nx = toByte(normal.x());
    const uint32_t ny = toByte(normal.y());
    const uint32_t nz = toByte(normal.z());
    return nx | (ny << 8) | (nz << 16) | (0xffu << 24);
}
}

namespace segmesh
{
bool loadObjMesh(const std::filesystem::path& filePath, CpuMesh& outMesh, std::string& error)
{
    outMesh = CpuMesh{};
    error.clear();

    OpenMeshTriMesh mesh;
    mesh.request_vertex_normals();
    mesh.request_face_normals();

    OpenMesh::IO::Options ioOptions;
    ioOptions += OpenMesh::IO::Options::VertexNormal;
    if (!OpenMesh::IO::read_mesh(mesh, filePath.string(), ioOptions))
    {
        error = "Unable to open OBJ file with OpenMesh: " + filePath.string();
        return false;
    }

    if (mesh.n_vertices() == 0 || mesh.n_faces() == 0)
    {
        error = "OBJ has no renderable geometry: " + filePath.string();
        return false;
    }

    if (!ioOptions.check(OpenMesh::IO::Options::VertexNormal))
    {
        mesh.update_face_normals();
        mesh.update_vertex_normals();
    }
    else
    {
        mesh.update_face_normals();
    }

    const std::size_t vertexCount = mesh.n_vertices();
    Eigen::MatrixXf positions(3, vertexCount);

    for (const OpenMeshTriMesh::VertexHandle vh : mesh.vertices())
    {
        positions.col(vh.idx()) = toEigen(mesh.point(vh));
    }

    const Eigen::Vector3f minPos = positions.rowwise().minCoeff();
    const Eigen::Vector3f maxPos = positions.rowwise().maxCoeff();
    const Eigen::Vector3f center = 0.5f * (minPos + maxPos);
    const float maxExtent = (maxPos - minPos).maxCoeff();
    const float scale = maxExtent > 1.0e-6f ? (2.0f / maxExtent) : 1.0f;

    positions.colwise() -= center;
    positions *= scale;

    std::vector<Eigen::Vector3f> preprocessedPositions(vertexCount);
    for (const OpenMeshTriMesh::VertexHandle vh : mesh.vertices())
    {
        preprocessedPositions[vh.idx()] = positions.col(vh.idx());
    }

    outMesh.vertices.reserve(vertexCount);
    std::vector<uint32_t> vertexMap(vertexCount, std::numeric_limits<uint32_t>::max());

    for (const OpenMeshTriMesh::VertexHandle vh : mesh.vertices())
    {
        const Eigen::Vector3f p = preprocessedPositions[vh.idx()];
        const Eigen::Vector3f n = normalizeSafe(toEigen(mesh.normal(vh)));

        MeshVertex vertex{};
        vertex.x = p.x();
        vertex.y = p.y();
        vertex.z = p.z();
        vertex.normal = packNormal(n);

        const uint32_t index = static_cast<uint32_t>(outMesh.vertices.size());
        outMesh.vertices.push_back(vertex);
        vertexMap[vh.idx()] = index;
    }

    outMesh.indices.reserve(mesh.n_faces() * 3);
    outMesh.faceCentroids.reserve(mesh.n_faces());
    outMesh.faceNormals.reserve(mesh.n_faces());
    outMesh.faceGaussianCurvatures.reserve(mesh.n_faces());
    outMesh.faceMeanCurvatures.reserve(mesh.n_faces());
    outMesh.faceAreas.reserve(mesh.n_faces());
    std::vector<int32_t> faceMap(mesh.n_faces(), -1);

    for (const OpenMeshTriMesh::FaceHandle fh : mesh.faces())
    {
        std::array<OpenMeshTriMesh::VertexHandle, 3> faceVertices = {
            OpenMeshTriMesh::VertexHandle(),
            OpenMeshTriMesh::VertexHandle(),
            OpenMeshTriMesh::VertexHandle(),
        };
        std::size_t corner = 0;
        for (auto fvIt = mesh.cfv_iter(fh); fvIt.is_valid(); ++fvIt)
        {
            if (corner < faceVertices.size())
            {
                faceVertices[corner] = *fvIt;
            }
            ++corner;
        }

        if (corner != 3)
        {
            continue;
        }

        const uint32_t i0 = vertexMap[faceVertices[0].idx()];
        const uint32_t i1 = vertexMap[faceVertices[1].idx()];
        const uint32_t i2 = vertexMap[faceVertices[2].idx()];
        const uint32_t outputFaceIndex = static_cast<uint32_t>(outMesh.faceCentroids.size());
        outMesh.indices.push_back(i0);
        outMesh.indices.push_back(i1);
        outMesh.indices.push_back(i2);

        const Eigen::Vector3f p0 = preprocessedPositions[faceVertices[0].idx()];
        const Eigen::Vector3f p1 = preprocessedPositions[faceVertices[1].idx()];
        const Eigen::Vector3f p2 = preprocessedPositions[faceVertices[2].idx()];
        const Eigen::Vector3f centroid = (p0 + p1 + p2) / 3.0f;
        const Eigen::Vector3f faceNormal = normalizeSafe(toEigen(mesh.normal(fh)));
        const float area = 0.5f * ((p1 - p0).cross(p2 - p0)).norm();

        outMesh.faceCentroids.push_back(toFloat3(centroid));
        outMesh.faceNormals.push_back(toFloat3(faceNormal));
        outMesh.faceAreas.push_back(area);
        faceMap[fh.idx()] = static_cast<int32_t>(outputFaceIndex);
    }

    outMesh.faceGaussianCurvatures.resize(outMesh.faceCentroids.size(), 0.0f);
    outMesh.faceMeanCurvatures.resize(outMesh.faceCentroids.size(), 0.0f);
    computeFaceCurvatures(
        mesh,
        preprocessedPositions,
        faceMap,
        outMesh.faceGaussianCurvatures,
        outMesh.faceMeanCurvatures
    );

    outMesh.faceAdjacency.resize(outMesh.faceCentroids.size());
    for (const OpenMeshTriMesh::FaceHandle fh : mesh.faces())
    {
        const int32_t outputFaceIndex = faceMap[fh.idx()];
        if (outputFaceIndex < 0)
        {
            continue;
        }

        FaceAdjacency adjacency{};
        std::size_t edgeIndex = 0;
        for (auto fhIt = mesh.cfh_iter(fh); fhIt.is_valid() && edgeIndex < adjacency.neighbors.size(); ++fhIt, ++edgeIndex)
        {
            const OpenMeshTriMesh::HalfedgeHandle heh = *fhIt;
            const OpenMeshTriMesh::HalfedgeHandle oppositeHeh = mesh.opposite_halfedge_handle(heh);
            const OpenMeshTriMesh::VertexHandle fromVh = mesh.from_vertex_handle(heh);
            const OpenMeshTriMesh::VertexHandle toVh = mesh.to_vertex_handle(heh);
            const Eigen::Vector3f edgeVector = preprocessedPositions[toVh.idx()] - preprocessedPositions[fromVh.idx()];
            adjacency.edgeLengths[edgeIndex] = edgeVector.norm();

            if (!mesh.is_boundary(oppositeHeh))
            {
                const OpenMeshTriMesh::FaceHandle neighborFh = mesh.face_handle(oppositeHeh);
                const int32_t neighborIndex = faceMap[neighborFh.idx()];
                adjacency.neighbors[edgeIndex] = neighborIndex;

                if (neighborIndex >= 0)
                {
                    const Eigen::Vector3f normal = normalizeSafe(toEigen(mesh.normal(fh)));
                    const Eigen::Vector3f neighborNormal = normalizeSafe(toEigen(mesh.normal(neighborFh)));
                    const float signedTurn = edgeVector.dot(normal.cross(neighborNormal));
                    adjacency.concavityScales[edgeIndex] = signedTurn < 0.0f ? 1.0f : 0.2f;
                }
            }
        }

        outMesh.faceAdjacency[static_cast<std::size_t>(outputFaceIndex)] = adjacency;
    }

    double graphicalDifferenceSum = 0.0;
    double engineeringNormalDifferenceSum = 0.0;
    double gaussianDifferenceSum = 0.0;
    double meanDifferenceSum = 0.0;
    uint32_t edgeCount = 0;
    for (std::size_t faceIndex = 0; faceIndex < outMesh.faceAdjacency.size(); ++faceIndex)
    {
        const FaceAdjacency& adjacency = outMesh.faceAdjacency[faceIndex];
        for (std::size_t edgeSlot = 0; edgeSlot < adjacency.neighbors.size(); ++edgeSlot)
        {
            const int32_t neighborIndex = adjacency.neighbors[edgeSlot];
            if (neighborIndex < 0 || faceIndex >= static_cast<std::size_t>(neighborIndex))
            {
                continue;
            }

            const double normalDot = std::clamp(
                static_cast<double>(outMesh.faceNormals[faceIndex].x) * static_cast<double>(outMesh.faceNormals[neighborIndex].x)
                    + static_cast<double>(outMesh.faceNormals[faceIndex].y) * static_cast<double>(outMesh.faceNormals[neighborIndex].y)
                    + static_cast<double>(outMesh.faceNormals[faceIndex].z) * static_cast<double>(outMesh.faceNormals[neighborIndex].z),
                -1.0,
                1.0
            );
            const double rawNormalDifference = 1.0 - normalDot;

            graphicalDifferenceSum += static_cast<double>(adjacency.concavityScales[edgeSlot]) * rawNormalDifference;
            engineeringNormalDifferenceSum += rawNormalDifference;
            gaussianDifferenceSum += std::abs(
                static_cast<double>(outMesh.faceGaussianCurvatures[faceIndex])
                    - static_cast<double>(outMesh.faceGaussianCurvatures[static_cast<std::size_t>(neighborIndex)])
            );
            meanDifferenceSum += std::abs(
                static_cast<double>(outMesh.faceMeanCurvatures[faceIndex])
                    - static_cast<double>(outMesh.faceMeanCurvatures[static_cast<std::size_t>(neighborIndex)])
            );
            ++edgeCount;
        }
    }

    if (edgeCount > 0)
    {
        const double inverseEdgeCount = 1.0 / static_cast<double>(edgeCount);
        outMesh.averageGraphicalDifference = static_cast<float>(std::max(graphicalDifferenceSum * inverseEdgeCount, 1.0e-12));
        outMesh.averageEngineeringNormalDifference =
            static_cast<float>(std::max(engineeringNormalDifferenceSum * inverseEdgeCount, 1.0e-12));
        outMesh.averageGaussianDifference = static_cast<float>(std::max(gaussianDifferenceSum * inverseEdgeCount, 1.0e-12));
        outMesh.averageMeanDifference = static_cast<float>(std::max(meanDifferenceSum * inverseEdgeCount, 1.0e-12));
    }

    if (outMesh.vertices.empty() || outMesh.indices.empty())
    {
        error = "OBJ has no renderable geometry: " + filePath.string();
        return false;
    }

    return true;
}

std::optional<std::filesystem::path> findFirstExisting(const std::vector<std::filesystem::path>& candidates)
{
    for (const std::filesystem::path& path : candidates)
    {
        if (std::filesystem::exists(path))
        {
            return path;
        }
    }

    return std::nullopt;
}

std::vector<std::filesystem::path> findObjFiles(const std::filesystem::path& directory)
{
    std::vector<std::filesystem::path> result;

    std::error_code ec;
    std::filesystem::directory_iterator it(directory, ec);
    if (ec)
    {
        return result;
    }

    const std::filesystem::directory_iterator end;
    for (; it != end; it.increment(ec))
    {
        if (ec)
        {
            break;
        }

        if (!it->is_regular_file(ec))
        {
            continue;
        }

        std::string extension = it->path().extension().string();
        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
        );

        if (extension == ".obj")
        {
            result.push_back(it->path());
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

int findModelIndex(const std::vector<std::filesystem::path>& models, const std::filesystem::path& target)
{
    for (std::size_t i = 0; i < models.size(); ++i)
    {
        std::error_code ec;
        if (std::filesystem::equivalent(models[i], target, ec) && !ec)
        {
            return static_cast<int>(i);
        }
    }

    return -1;
}
}
