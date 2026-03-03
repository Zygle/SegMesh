#include "obj_loader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace
{
using segmesh::Float3;
using segmesh::MeshVertex;

Float3 add(const Float3& a, const Float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Float3 sub(const Float3& a, const Float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Float3 cross(const Float3& a, const Float3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

Float3 normalize(const Float3& v, const Float3& fallback = {0.0f, 1.0f, 0.0f})
{
    const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (lenSq < 1.0e-12f)
    {
        return fallback;
    }

    const float invLen = 1.0f / std::sqrt(lenSq);
    return {v.x * invLen, v.y * invLen, v.z * invLen};
}

uint32_t packNormal(const Float3& normal)
{
    auto toByte = [](float value) -> uint8_t
    {
        const float normalized = std::clamp(value * 0.5f + 0.5f, 0.0f, 1.0f);
        return static_cast<uint8_t>(normalized * 255.0f + 0.5f);
    };

    const uint32_t nx = toByte(normal.x);
    const uint32_t ny = toByte(normal.y);
    const uint32_t nz = toByte(normal.z);
    return nx | (ny << 8) | (nz << 16) | (0xffu << 24);
}

struct VertexKey
{
    int position = -1;
    int normal = -1;

    bool operator==(const VertexKey& rhs) const
    {
        return position == rhs.position && normal == rhs.normal;
    }
};

struct VertexKeyHash
{
    std::size_t operator()(const VertexKey& key) const
    {
        const std::size_t p = static_cast<std::size_t>(key.position + 1);
        const std::size_t n = static_cast<std::size_t>(key.normal + 2);
        return (p * 73856093u) ^ (n * 19349663u);
    }
};

bool parseFaceToken(const std::string& token, int& outPosition, int& outNormal)
{
    outPosition = 0;
    outNormal = 0;

    try
    {
        const std::size_t firstSlash = token.find('/');
        if (firstSlash == std::string::npos)
        {
            outPosition = std::stoi(token);
            return true;
        }

        outPosition = std::stoi(token.substr(0, firstSlash));

        const std::size_t secondSlash = token.find('/', firstSlash + 1);
        if (secondSlash == std::string::npos)
        {
            return true;
        }

        if (secondSlash + 1 < token.size())
        {
            outNormal = std::stoi(token.substr(secondSlash + 1));
        }

        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool resolveObjIndex(int rawIndex, std::size_t count, int& outIndex)
{
    if (rawIndex > 0)
    {
        outIndex = rawIndex - 1;
    }
    else if (rawIndex < 0)
    {
        outIndex = static_cast<int>(count) + rawIndex;
    }
    else
    {
        return false;
    }

    return outIndex >= 0 && outIndex < static_cast<int>(count);
}
} // namespace

namespace segmesh
{
bool loadObjMesh(const std::filesystem::path& filePath, CpuMesh& outMesh, std::string& error)
{
    std::ifstream input(filePath);
    if (!input)
    {
        error = "Unable to open OBJ file: " + filePath.string();
        return false;
    }

    std::vector<Float3> positions;
    std::vector<Float3> normals;

    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> vertexMap;
    std::vector<Float3> uniquePositions;
    std::vector<Float3> accumulatedNormals;
    std::vector<bool> hasImportedNormal;

    auto makeVertex = [&](int positionIndex, int normalIndex) -> uint32_t
    {
        const VertexKey key{positionIndex, normalIndex};
        const auto it = vertexMap.find(key);
        if (it != vertexMap.end())
        {
            return it->second;
        }

        const Float3& p = positions[static_cast<std::size_t>(positionIndex)];
        const bool hasNormal = normalIndex >= 0;
        const Float3 normal = hasNormal
            ? normalize(normals[static_cast<std::size_t>(normalIndex)])
            : Float3{0.0f, 1.0f, 0.0f};

        MeshVertex v{};
        v.x = p.x;
        v.y = p.y;
        v.z = p.z;
        v.normal = packNormal(normal);

        const uint32_t newIndex = static_cast<uint32_t>(outMesh.vertices.size());
        outMesh.vertices.push_back(v);
        uniquePositions.push_back(p);
        accumulatedNormals.push_back({0.0f, 0.0f, 0.0f});
        hasImportedNormal.push_back(hasNormal);
        vertexMap.emplace(key, newIndex);
        return newIndex;
    };

    std::string line;
    uint32_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;

        const std::size_t commentPos = line.find('#');
        if (commentPos != std::string::npos)
        {
            line.erase(commentPos);
        }

        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if (type.empty())
        {
            continue;
        }

        if (type == "v")
        {
            Float3 p{};
            if (!(iss >> p.x >> p.y >> p.z))
            {
                error = "Malformed vertex at line " + std::to_string(lineNumber);
                return false;
            }
            positions.push_back(p);
            continue;
        }

        if (type == "vn")
        {
            Float3 n{};
            if (!(iss >> n.x >> n.y >> n.z))
            {
                error = "Malformed normal at line " + std::to_string(lineNumber);
                return false;
            }
            normals.push_back(n);
            continue;
        }

        if (type != "f")
        {
            continue;
        }

        std::vector<std::pair<int, int> > face;
        std::string token;
        while (iss >> token)
        {
            int rawPosition = 0;
            int rawNormal = 0;
            if (!parseFaceToken(token, rawPosition, rawNormal))
            {
                error = "Malformed face token at line " + std::to_string(lineNumber) + ": " + token;
                return false;
            }

            int positionIndex = -1;
            if (!resolveObjIndex(rawPosition, positions.size(), positionIndex))
            {
                error = "Face position index out of range at line " + std::to_string(lineNumber);
                return false;
            }

            int normalIndex = -1;
            if (rawNormal != 0 && !resolveObjIndex(rawNormal, normals.size(), normalIndex))
            {
                error = "Face normal index out of range at line " + std::to_string(lineNumber);
                return false;
            }

            face.emplace_back(positionIndex, normalIndex);
        }

        if (face.size() < 3)
        {
            continue;
        }

        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            const uint32_t i0 = makeVertex(face[0].first, face[0].second);
            const uint32_t i1 = makeVertex(face[i].first, face[i].second);
            const uint32_t i2 = makeVertex(face[i + 1].first, face[i + 1].second);

            outMesh.indices.push_back(i0);
            outMesh.indices.push_back(i1);
            outMesh.indices.push_back(i2);

            const Float3& p0 = uniquePositions[static_cast<std::size_t>(i0)];
            const Float3& p1 = uniquePositions[static_cast<std::size_t>(i1)];
            const Float3& p2 = uniquePositions[static_cast<std::size_t>(i2)];
            const Float3 faceNormal = normalize(cross(sub(p1, p0), sub(p2, p0)));

            if (!hasImportedNormal[static_cast<std::size_t>(i0)])
            {
                accumulatedNormals[static_cast<std::size_t>(i0)] = add(accumulatedNormals[static_cast<std::size_t>(i0)], faceNormal);
            }
            if (!hasImportedNormal[static_cast<std::size_t>(i1)])
            {
                accumulatedNormals[static_cast<std::size_t>(i1)] = add(accumulatedNormals[static_cast<std::size_t>(i1)], faceNormal);
            }
            if (!hasImportedNormal[static_cast<std::size_t>(i2)])
            {
                accumulatedNormals[static_cast<std::size_t>(i2)] = add(accumulatedNormals[static_cast<std::size_t>(i2)], faceNormal);
            }
        }
    }

    if (outMesh.vertices.empty() || outMesh.indices.empty())
    {
        error = "OBJ has no renderable geometry: " + filePath.string();
        return false;
    }

    for (std::size_t i = 0; i < outMesh.vertices.size(); ++i)
    {
        if (!hasImportedNormal[i])
        {
            const Float3 n = normalize(accumulatedNormals[i]);
            outMesh.vertices[i].normal = packNormal(n);
        }
    }

    Float3 minPos{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    Float3 maxPos{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
    };

    for (const MeshVertex& v : outMesh.vertices)
    {
        minPos.x = std::min(minPos.x, v.x);
        minPos.y = std::min(minPos.y, v.y);
        minPos.z = std::min(minPos.z, v.z);

        maxPos.x = std::max(maxPos.x, v.x);
        maxPos.y = std::max(maxPos.y, v.y);
        maxPos.z = std::max(maxPos.z, v.z);
    }

    const Float3 center{
        (minPos.x + maxPos.x) * 0.5f,
        (minPos.y + maxPos.y) * 0.5f,
        (minPos.z + maxPos.z) * 0.5f,
    };

    const float maxExtent = std::max({maxPos.x - minPos.x, maxPos.y - minPos.y, maxPos.z - minPos.z});
    const float scale = maxExtent > 1.0e-6f ? (2.0f / maxExtent) : 1.0f;

    for (MeshVertex& v : outMesh.vertices)
    {
        v.x = (v.x - center.x) * scale;
        v.y = (v.y - center.y) * scale;
        v.z = (v.z - center.z) * scale;
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
