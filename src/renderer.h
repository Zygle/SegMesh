#pragma once

#include "mesh_types.h"
#include "ui_panel.h"

#include <bgfx/bgfx.h>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <string>

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
    void resize(uint32_t width, uint32_t height);
    void renderScene(uint32_t width, uint32_t height, float modelRotation, const RendererUiState& uiState);
    void frame();
    const char* rendererName() const;
    void shutdown();

private:
    bool uploadMesh(const CpuMesh& mesh, GpuMesh& outMesh, std::string& error);
    void destroyGpuMesh();

    bool initialized_ = false;

    GpuMesh gpuMesh_{};
    bgfx::ProgramHandle meshProgram_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uBaseColor_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uLightDir_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uLightColor_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uCameraPos_ = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle uMaterial_ = BGFX_INVALID_HANDLE;
};
}
