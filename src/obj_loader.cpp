#include "obj_loader.h"

#include <Eigen/Dense>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{
using OpenMeshTriMesh = OpenMesh::TriMesh_ArrayKernelT<>;
using segmesh::Float3;
using segmesh::MeshVertex;

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
    outMesh.faceAreas.reserve(mesh.n_faces());

    for (const OpenMeshTriMesh::FaceHandle fh : mesh.faces())
    {
        std::array<OpenMeshTriMesh::VertexHandle, 3> faceVertices{};
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
        outMesh.indices.push_back(i0);
        outMesh.indices.push_back(i1);
        outMesh.indices.push_back(i2);

        const Eigen::Vector3f p0 = preprocessedPositions[faceVertices[0].idx()];
        const Eigen::Vector3f p1 = preprocessedPositions[faceVertices[1].idx()];
        const Eigen::Vector3f p2 = preprocessedPositions[faceVertices[2].idx()];
        const Eigen::Vector3f centroid = (p0 + p1 + p2) / 3.0f;
        const float area = 0.5f * ((p1 - p0).cross(p2 - p0)).norm();

        outMesh.faceCentroids.push_back(toFloat3(centroid));
        outMesh.faceAreas.push_back(area);
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
