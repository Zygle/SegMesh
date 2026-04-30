#pragma once

#include <bgfx/bgfx.h>

#include <array>
#include <cstdint>
#include <vector>

namespace segmesh
{
enum class SegmentationModelType : int
{
    Graphical = 0,
    Engineering = 1,
};

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

struct FaceRenderEdges
{
    std::array<std::array<uint32_t, 2>, 3> indices = {{{0, 0}, {0, 0}, {0, 0}}};
};

struct CpuMesh
{
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Float3> faceCentroids;
    std::vector<Float3> faceNormals;
    std::vector<float> faceGaussianCurvatures;
    std::vector<float> faceMeanCurvatures;
    std::vector<float> faceAreas;
    std::vector<FaceAdjacency> faceAdjacency;
    std::vector<FaceRenderEdges> faceRenderEdges;
    float averageGraphicalDifference = 1.0f;
    float averageEngineeringNormalDifference = 1.0f;
    float averageGaussianDifference = 1.0f;
    float averageMeanDifference = 1.0f;
};

struct GpuMesh
{
    bgfx::VertexBufferHandle vbh = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle ibh = BGFX_INVALID_HANDLE;
    uint32_t indexCount = 0;
};
}
