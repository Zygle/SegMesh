#pragma once

#include <bgfx/bgfx.h>

#include <array>
#include <cstdint>
#include <vector>

namespace segmesh
{
struct Float3
{
    float x;
    float y;
    float z;
};

struct MeshVertex
{
    float x;
    float y;
    float z;
    uint32_t normal;

    static bgfx::VertexLayout layout;

    static void initLayout();
};

struct FaceAdjacency
{
    std::array<int32_t, 3> neighbors = {-1, -1, -1};
    std::array<float, 3> edgeLengths = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> concavityScales = {0.2f, 0.2f, 0.2f};
};

struct CpuMesh
{
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Float3> faceCentroids;
    std::vector<Float3> faceNormals;
    std::vector<float> faceAreas;
    std::vector<FaceAdjacency> faceAdjacency;
};

struct GpuMesh
{
    bgfx::VertexBufferHandle vbh = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle ibh = BGFX_INVALID_HANDLE;
    uint32_t indexCount = 0;
};
}
