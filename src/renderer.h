#pragma once

#include "mesh_types.h"
#include "ui.h"

#include <bgfx/bgfx.h>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <string>
#include <vector>

namespace segmesh
{
class Renderer
{
public:
    Renderer() = default;
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    ~Renderer();

    bool initialize(GLFWwindow* window, uint32_t width, uint32_t height, std::string& error);
    bool loadMesh(const CpuMesh& mesh, std::string& error);
    bool setSeedTriangle(const CpuMesh& mesh, uint32_t triangleIndex, std::string& error);
    bool setTriangleGroups(
        const CpuMesh& mesh,
        const std::vector<uint32_t>& triangleGroups,
        uint32_t groupCount,
        std::string& error
    );
    void clearSeedTriangle();
    void clearTriangleGroups();
    void resize(uint32_t width, uint32_t height);
    void renderScene(uint32_t width, uint32_t height, float modelRotation, const RendererUiState& uiState);
    void frame();
    const char* rendererName() const;
    void shutdown();

private:
    bool uploadMesh(const CpuMesh& mesh, GpuMesh& outMesh, std::string& error);
    void destroyGpuMesh();

    bool initialized_ = false;

    struct TriangleGroupDraw
    {
        bgfx::IndexBufferHandle ibh = BGFX_INVALID_HANDLE;
        uint32_t indexCount = 0;
        Float3 color{0.0f, 0.0f, 0.0f};
    };

    GpuMesh gpuMesh_{};
    bgfx::ProgramHandle meshProgram_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uBaseColor_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uLightDir_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uLightColor_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uCameraPos_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uMaterial_ = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle seedTriangleIbh_ = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle segmentBorderIbh_ = BGFX_INVALID_HANDLE;
    uint32_t segmentBorderIndexCount_ = 0;
    std::vector<uint32_t> seedTriangleIndices_;
    std::vector<TriangleGroupDraw> triangleGroupDraws_;
};
}
