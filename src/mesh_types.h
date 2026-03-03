#pragma once

#include <bgfx/bgfx.h>

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

struct CpuMesh
{
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;
};

struct GpuMesh
{
    bgfx::VertexBufferHandle vbh = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle ibh = BGFX_INVALID_HANDLE;
    uint32_t indexCount = 0;
};
}
