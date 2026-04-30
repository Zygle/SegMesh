#include "renderer.h"

#include <bgfx/platform.h>
#include <GLFW/glfw3native.h>
#include <bx/math.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{
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

std::optional<std::vector<uint8_t> > readBinaryFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        return std::nullopt;
    }

    const std::streamsize size = input.tellg();
    if (size <= 0)
    {
        return std::nullopt;
    }

    std::vector<uint8_t> bytes(static_cast<std::size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), size))
    {
        return std::nullopt;
    }

    return bytes;
}

bgfx::ShaderHandle loadShaderFromFile(const std::filesystem::path& path)
{
    const auto bytes = readBinaryFile(path);
    if (!bytes.has_value())
    {
        return BGFX_INVALID_HANDLE;
    }

    const bgfx::Memory* memory = bgfx::copy(bytes->data(), static_cast<uint32_t>(bytes->size()));
    bgfx::ShaderHandle shader = bgfx::createShader(memory);
    if (bgfx::isValid(shader))
    {
        const std::string name = path.filename().string();
        bgfx::setName(shader, name.c_str());
    }

    return shader;
}

segmesh::Float3 normalize(const segmesh::Float3& v, const segmesh::Float3& fallback)
{
    const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (lenSq < 1.0e-12f)
    {
        return fallback;
    }

    const float invLen = 1.0f / std::sqrt(lenSq);
    return {v.x * invLen, v.y * invLen, v.z * invLen};
}

void destroyUniform(bgfx::UniformHandle& uniform)
{
    if (bgfx::isValid(uniform))
    {
        bgfx::destroy(uniform);
    }

    uniform = BGFX_INVALID_HANDLE;
}

void destroyGpuMeshHandles(segmesh::GpuMesh& gpuMesh)
{
    if (bgfx::isValid(gpuMesh.vbh))
    {
        bgfx::destroy(gpuMesh.vbh);
    }

    if (bgfx::isValid(gpuMesh.ibh))
    {
        bgfx::destroy(gpuMesh.ibh);
    }

    gpuMesh.vbh = BGFX_INVALID_HANDLE;
    gpuMesh.ibh = BGFX_INVALID_HANDLE;
    gpuMesh.indexCount = 0;
}

void destroyIndexBuffer(bgfx::IndexBufferHandle& handle)
{
    if (bgfx::isValid(handle))
    {
        bgfx::destroy(handle);
    }

    handle = BGFX_INVALID_HANDLE;
}

segmesh::Float3 colorForGroup(uint32_t groupIndex)
{
    static constexpr std::array<segmesh::Float3, 32> kGroupPalette = {{
        {0.90f, 0.32f, 0.32f},
        {0.24f, 0.62f, 0.90f},
        {0.34f, 0.76f, 0.42f},
        {0.94f, 0.72f, 0.24f},
        {0.62f, 0.42f, 0.86f},
        {0.24f, 0.72f, 0.78f},
        {0.87f, 0.42f, 0.68f},
        {0.54f, 0.70f, 0.25f},
        {0.95f, 0.48f, 0.22f},
        {0.42f, 0.50f, 0.92f},
        {0.28f, 0.80f, 0.62f},
        {0.82f, 0.36f, 0.88f},
        {0.74f, 0.78f, 0.26f},
        {0.20f, 0.54f, 0.70f},
        {0.96f, 0.58f, 0.55f},
        {0.42f, 0.84f, 0.30f},
        {0.76f, 0.46f, 0.28f},
        {0.52f, 0.38f, 0.68f},
        {0.28f, 0.68f, 0.50f},
        {0.92f, 0.82f, 0.40f},
        {0.30f, 0.42f, 0.74f},
        {0.82f, 0.30f, 0.46f},
        {0.40f, 0.76f, 0.88f},
        {0.64f, 0.58f, 0.24f},
        {0.98f, 0.38f, 0.28f},
        {0.48f, 0.64f, 0.34f},
        {0.72f, 0.36f, 0.64f},
        {0.26f, 0.76f, 0.70f},
        {0.70f, 0.54f, 0.88f},
        {0.88f, 0.64f, 0.34f},
        {0.34f, 0.58f, 0.40f},
        {0.78f, 0.28f, 0.72f},
    }};

    return kGroupPalette[static_cast<std::size_t>(groupIndex % kGroupPalette.size())];
}

std::vector<uint32_t> buildSegmentBorderIndices(
    const segmesh::CpuMesh& mesh,
    const std::vector<uint32_t>& triangleGroups
)
{
    const uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    std::vector<uint32_t> segmentBorderIndices;
    segmentBorderIndices.reserve(triangleCount * 2);

    if (mesh.faceAdjacency.size() != triangleCount || mesh.faceRenderEdges.size() != triangleCount)
    {
        return segmentBorderIndices;
    }

    for (uint32_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex)
    {
        const uint32_t groupIndex = triangleGroups[static_cast<std::size_t>(triangleIndex)];

        const segmesh::FaceAdjacency& adjacency = mesh.faceAdjacency[static_cast<std::size_t>(triangleIndex)];
        const segmesh::FaceRenderEdges& renderEdges = mesh.faceRenderEdges[static_cast<std::size_t>(triangleIndex)];
        for (std::size_t edgeSlot = 0; edgeSlot < adjacency.neighbors.size(); ++edgeSlot)
        {
            const int32_t neighborIndex = adjacency.neighbors[edgeSlot];
            if (neighborIndex < 0 || triangleIndex >= static_cast<uint32_t>(neighborIndex))
            {
                continue;
            }

            const uint32_t neighborGroupIndex = triangleGroups[static_cast<std::size_t>(neighborIndex)];
            if (groupIndex != neighborGroupIndex)
            {
                segmentBorderIndices.push_back(renderEdges.indices[edgeSlot][0]);
                segmentBorderIndices.push_back(renderEdges.indices[edgeSlot][1]);
            }
        }
    }

    return segmentBorderIndices;
}
}

namespace segmesh
{
Renderer::~Renderer()
{
    shutdown();
}

bool Renderer::initialize(GLFWwindow* window, uint32_t width, uint32_t height, std::string& error)
{
    shutdown();

    if (window == nullptr)
    {
        error = "Renderer initialization received null window.";
        return false;
    }

    bgfx::PlatformData platformData{};
    platformData.context = nullptr;
    platformData.backBuffer = nullptr;
    platformData.backBufferDS = nullptr;
    bool detectedNativeBackend = false;

#if defined(GLFW_EXPOSE_NATIVE_WAYLAND)
    if (!detectedNativeBackend)
    {
        wl_display* display = glfwGetWaylandDisplay();
        wl_surface* surface = glfwGetWaylandWindow(window);
        if (display != nullptr && surface != nullptr)
        {
            platformData.ndt = display;
            platformData.nwh = surface;
            platformData.type = bgfx::NativeWindowHandleType::Wayland;
            detectedNativeBackend = true;
        }
    }
#endif

#if defined(GLFW_EXPOSE_NATIVE_X11)
    if (!detectedNativeBackend)
    {
        Display* display = glfwGetX11Display();
        const Window x11Window = glfwGetX11Window(window);
        if (display != nullptr && x11Window != 0)
        {
            platformData.ndt = display;
            platformData.nwh = reinterpret_cast<void*>(static_cast<uintptr_t>(x11Window));
            // bgfx treats X11 as the Linux default native handle type.
            platformData.type = bgfx::NativeWindowHandleType::Default;
            detectedNativeBackend = true;
        }
    }
#endif

    if (!detectedNativeBackend)
    {
        error = "Unable to detect a supported native window backend. Build with Wayland and/or X11 support.";
        return false;
    }

    bgfx::renderFrame();

    bgfx::Init init{};
    init.type = bgfx::RendererType::Vulkan;
    init.platformData = platformData;
    init.resolution.width = width;
    init.resolution.height = height;
    init.resolution.reset = BGFX_RESET_VSYNC;

    if (!bgfx::init(init))
    {
        error = "bgfx::init failed";
        return false;
    }
    initialized_ = true;

    const auto shaderDir = findFirstExisting({
        "build/shaders/spirv",
        "shaders/spirv",
        "../build/shaders/spirv",
        "../../build/shaders/spirv",
    });
    if (!shaderDir.has_value())
    {
        error = "Unable to find bgfx runtime shader directory.";
        shutdown();
        return false;
    }

    bgfx::ShaderHandle vsh = loadShaderFromFile(*shaderDir / "vs_triangle.bin");
    bgfx::ShaderHandle fsh = loadShaderFromFile(*shaderDir / "fs_triangle.bin");
    if (!bgfx::isValid(vsh) || !bgfx::isValid(fsh))
    {
        if (bgfx::isValid(vsh))
        {
            bgfx::destroy(vsh);
        }
        if (bgfx::isValid(fsh))
        {
            bgfx::destroy(fsh);
        }
        error = "Failed to load shader binaries from: " + shaderDir->string();
        shutdown();
        return false;
    }

    meshProgram_ = bgfx::createProgram(vsh, fsh, true);
    if (!bgfx::isValid(meshProgram_))
    {
        error = "Failed to create mesh program.";
        shutdown();
        return false;
    }

    MeshVertex::initLayout();

    uBaseColor_ = bgfx::createUniform("u_baseColor", bgfx::UniformType::Vec4);
    uLightDir_ = bgfx::createUniform("u_lightDir", bgfx::UniformType::Vec4);
    uLightColor_ = bgfx::createUniform("u_lightColor", bgfx::UniformType::Vec4);
    uCameraPos_ = bgfx::createUniform("u_cameraPos", bgfx::UniformType::Vec4);
    uMaterial_ = bgfx::createUniform("u_material", bgfx::UniformType::Vec4);

    if (!bgfx::isValid(uBaseColor_) || !bgfx::isValid(uLightDir_) || !bgfx::isValid(uLightColor_)
        || !bgfx::isValid(uCameraPos_) || !bgfx::isValid(uMaterial_))
    {
        error = "Failed to create one or more renderer uniforms.";
        shutdown();
        return false;
    }

    return true;
}

bool Renderer::uploadMesh(const CpuMesh& mesh, GpuMesh& outMesh, std::string& error)
{
    const bgfx::Memory* vmem = bgfx::copy(
        mesh.vertices.data(),
        static_cast<uint32_t>(mesh.vertices.size() * sizeof(MeshVertex))
    );
    outMesh.vbh = bgfx::createVertexBuffer(vmem, MeshVertex::layout);

    const bgfx::Memory* imem = bgfx::copy(
        mesh.indices.data(),
        static_cast<uint32_t>(mesh.indices.size() * sizeof(uint32_t))
    );
    outMesh.ibh = bgfx::createIndexBuffer(imem, BGFX_BUFFER_INDEX32);
    outMesh.indexCount = static_cast<uint32_t>(mesh.indices.size());

    if (!bgfx::isValid(outMesh.vbh) || !bgfx::isValid(outMesh.ibh))
    {
        error = "Failed to create bgfx mesh buffers.";
        destroyGpuMeshHandles(outMesh);
        return false;
    }

    return true;
}

bool Renderer::loadMesh(const CpuMesh& mesh, std::string& error)
{
    if (!initialized_)
    {
        error = "Renderer is not initialized.";
        return false;
    }

    GpuMesh newMesh{};
    if (!uploadMesh(mesh, newMesh, error))
    {
        return false;
    }

    destroyGpuMesh();
    gpuMesh_ = newMesh;
    clearTriangleGroups();
    clearSeedTriangle();
    return true;
}

bool Renderer::setSeedTriangle(const CpuMesh& mesh, uint32_t triangleIndex, std::string& error)
{
    if (!initialized_ || !bgfx::isValid(gpuMesh_.vbh) || !bgfx::isValid(gpuMesh_.ibh))
    {
        error = "Renderer mesh must be loaded before selecting a seed triangle.";
        return false;
    }

    const uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (triangleIndex >= triangleCount)
    {
        error = "Seed triangle index is out of range.";
        return false;
    }

    const uint32_t offset = triangleIndex * 3;
    std::vector<uint32_t> newSeedTriangleIndices = seedTriangleIndices_;
    newSeedTriangleIndices.push_back(mesh.indices[offset + 0]);
    newSeedTriangleIndices.push_back(mesh.indices[offset + 1]);
    newSeedTriangleIndices.push_back(mesh.indices[offset + 2]);

    const bgfx::Memory* imem = bgfx::copy(
        newSeedTriangleIndices.data(),
        static_cast<uint32_t>(newSeedTriangleIndices.size() * sizeof(uint32_t))
    );
    const bgfx::IndexBufferHandle ibh = bgfx::createIndexBuffer(imem, BGFX_BUFFER_INDEX32);
    if (!bgfx::isValid(ibh))
    {
        error = "Failed to create seed triangle index buffer.";
        return false;
    }

    if (bgfx::isValid(seedTriangleIbh_))
    {
        bgfx::destroy(seedTriangleIbh_);
    }
    seedTriangleIbh_ = ibh;
    seedTriangleIndices_ = std::move(newSeedTriangleIndices);
    return true;
}

bool Renderer::setTriangleGroups(
    const CpuMesh& mesh,
    const std::vector<uint32_t>& triangleGroups,
    uint32_t groupCount,
    std::string& error
)
{
    if (!initialized_ || !bgfx::isValid(gpuMesh_.vbh) || !bgfx::isValid(gpuMesh_.ibh))
    {
        error = "Renderer mesh must be loaded before assigning triangle groups.";
        return false;
    }

    const uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (groupCount == 0)
    {
        error = "Triangle group count must be greater than zero.";
        return false;
    }

    if (triangleGroups.size() != triangleCount)
    {
        error = "Triangle group assignment size does not match the triangle count.";
        return false;
    }

    std::vector<std::vector<uint32_t> > groupedIndices(groupCount);
    for (uint32_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex)
    {
        const uint32_t groupIndex = triangleGroups[triangleIndex];
        if (groupIndex >= groupCount)
        {
            error = "Triangle group assignment contains an out-of-range group id.";
            return false;
        }

        const uint32_t indexOffset = triangleIndex * 3;
        std::vector<uint32_t>& groupIndices = groupedIndices[groupIndex];
        groupIndices.push_back(mesh.indices[indexOffset + 0]);
        groupIndices.push_back(mesh.indices[indexOffset + 1]);
        groupIndices.push_back(mesh.indices[indexOffset + 2]);
    }

    std::vector<TriangleGroupDraw> newTriangleGroupDraws;
    newTriangleGroupDraws.reserve(groupCount);
    for (uint32_t groupIndex = 0; groupIndex < groupCount; ++groupIndex)
    {
        const std::vector<uint32_t>& groupIndices = groupedIndices[groupIndex];
        if (groupIndices.empty())
        {
            continue;
        }

        const bgfx::Memory* imem = bgfx::copy(
            groupIndices.data(),
            static_cast<uint32_t>(groupIndices.size() * sizeof(uint32_t))
        );
        TriangleGroupDraw groupDraw{};
        groupDraw.ibh = bgfx::createIndexBuffer(imem, BGFX_BUFFER_INDEX32);
        groupDraw.indexCount = static_cast<uint32_t>(groupIndices.size());
        groupDraw.color = colorForGroup(groupIndex);
        if (!bgfx::isValid(groupDraw.ibh))
        {
            error = "Failed to create one of the triangle group index buffers.";
            for (TriangleGroupDraw& createdGroupDraw : newTriangleGroupDraws)
            {
                destroyIndexBuffer(createdGroupDraw.ibh);
            }
            return false;
        }

        newTriangleGroupDraws.push_back(groupDraw);
    }

    const std::vector<uint32_t> segmentBorderIndices = buildSegmentBorderIndices(mesh, triangleGroups);

    bgfx::IndexBufferHandle newSegmentBorderIbh = BGFX_INVALID_HANDLE;
    uint32_t newSegmentBorderIndexCount = 0;
    if (!segmentBorderIndices.empty())
    {
        const bgfx::Memory* imem = bgfx::copy(
            segmentBorderIndices.data(),
            static_cast<uint32_t>(segmentBorderIndices.size() * sizeof(uint32_t))
        );
        newSegmentBorderIbh = bgfx::createIndexBuffer(imem, BGFX_BUFFER_INDEX32);
        if (!bgfx::isValid(newSegmentBorderIbh))
        {
            error = "Failed to create the segment border index buffer.";
            for (TriangleGroupDraw& createdGroupDraw : newTriangleGroupDraws)
            {
                destroyIndexBuffer(createdGroupDraw.ibh);
            }
            return false;
        }

        newSegmentBorderIndexCount = static_cast<uint32_t>(segmentBorderIndices.size());
    }

    clearTriangleGroups();
    segmentBorderIbh_ = newSegmentBorderIbh;
    segmentBorderIndexCount_ = newSegmentBorderIndexCount;
    triangleGroupDraws_ = std::move(newTriangleGroupDraws);
    return true;
}

void Renderer::clearSeedTriangle()
{
    destroyIndexBuffer(seedTriangleIbh_);
    seedTriangleIndices_.clear();
}

void Renderer::clearTriangleGroups()
{
    destroyIndexBuffer(segmentBorderIbh_);
    segmentBorderIndexCount_ = 0;

    for (TriangleGroupDraw& groupDraw : triangleGroupDraws_)
    {
        destroyIndexBuffer(groupDraw.ibh);
    }

    triangleGroupDraws_.clear();
}

void Renderer::resize(uint32_t width, uint32_t height)
{
    if (!initialized_ || width == 0 || height == 0)
    {
        return;
    }

    bgfx::reset(width, height, BGFX_RESET_VSYNC);
}

void Renderer::renderScene(uint32_t width, uint32_t height, float modelRotation, const RendererUiState& uiState)
{
    if (!initialized_ || !bgfx::isValid(meshProgram_) || !bgfx::isValid(gpuMesh_.vbh) || !bgfx::isValid(gpuMesh_.ibh))
    {
        return;
    }

    const bgfx::Caps* caps = bgfx::getCaps();

    bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x303040ff, 1.0f, 0);
    bgfx::setViewRect(0, 0, 0, static_cast<uint16_t>(width), static_cast<uint16_t>(height));

    const float eyeX = std::sin(uiState.cameraYaw) * std::cos(uiState.cameraPitch) * uiState.cameraDistance;
    const float eyeY = std::sin(uiState.cameraPitch) * uiState.cameraDistance;
    const float eyeZ = std::cos(uiState.cameraYaw) * std::cos(uiState.cameraPitch) * uiState.cameraDistance;

    const bx::Vec3 eye{eyeX, eyeY, eyeZ};
    const bx::Vec3 at{0.0f, 0.0f, 0.0f};
    float view[16];
    bx::mtxLookAt(view, eye, at);

    float proj[16];
    bx::mtxProj(proj, 60.0f, static_cast<float>(width) / static_cast<float>(height), 0.01f, 100.0f, caps->homogeneousDepth);
    bgfx::setViewTransform(0, view, proj);

    float model[16];
    bx::mtxRotateY(model, modelRotation);

    const Float3 lightDir = normalize(
        {uiState.lightDirection[0], uiState.lightDirection[1], uiState.lightDirection[2]},
        {-0.45f, -0.9f, -0.25f}
    );
    const float baseColorUniform[4] = {uiState.baseColor[0], uiState.baseColor[1], uiState.baseColor[2], 1.0f};
    const float lightDirUniform[4] = {lightDir.x, lightDir.y, lightDir.z, uiState.lightIntensity};
    const float lightColorUniform[4] = {uiState.lightColor[0], uiState.lightColor[1], uiState.lightColor[2], 1.0f};
    const float cameraPosUniform[4] = {eyeX, eyeY, eyeZ, 1.0f};
    const float materialUniform[4] = {uiState.ambientStrength, uiState.specularStrength, uiState.shininess, 0.0f};

    const uint64_t state = BGFX_STATE_WRITE_RGB
        | BGFX_STATE_WRITE_A
        | BGFX_STATE_WRITE_Z
        | BGFX_STATE_DEPTH_TEST_LESS
        | BGFX_STATE_MSAA;

    // Reuse the same shaded draw setup for the base mesh, groups, outlines, and seed overlays.
    const auto submitMeshPass =
        [&](bgfx::IndexBufferHandle ibh, uint32_t indexCount, const float color[4], const float material[4], uint64_t drawState)
    {
        bgfx::setTransform(model);
        bgfx::setVertexBuffer(0, gpuMesh_.vbh);
        bgfx::setIndexBuffer(ibh, 0, indexCount);
        bgfx::setUniform(uBaseColor_, color);
        bgfx::setUniform(uLightDir_, lightDirUniform);
        bgfx::setUniform(uLightColor_, lightColorUniform);
        bgfx::setUniform(uCameraPos_, cameraPosUniform);
        bgfx::setUniform(uMaterial_, material);
        bgfx::setState(drawState);
        bgfx::submit(0, meshProgram_);
    };

    if (triangleGroupDraws_.empty())
    {
        submitMeshPass(gpuMesh_.ibh, gpuMesh_.indexCount, baseColorUniform, materialUniform, state);
    }
    else
    {
        for (const TriangleGroupDraw& groupDraw : triangleGroupDraws_)
        {
            const float groupColorUniform[4] = {groupDraw.color.x, groupDraw.color.y, groupDraw.color.z, 1.0f};
            submitMeshPass(groupDraw.ibh, groupDraw.indexCount, groupColorUniform, materialUniform, state);
        }
    }

    if (uiState.showSegmentBorders && bgfx::isValid(segmentBorderIbh_) && segmentBorderIndexCount_ > 0)
    {
        const float borderColorUniform[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        const float borderMaterialUniform[4] = {1.0f, 0.0f, 1.0f, 0.0f};
        const uint64_t borderState = BGFX_STATE_WRITE_RGB
            | BGFX_STATE_WRITE_A
            | BGFX_STATE_DEPTH_TEST_LEQUAL
            | BGFX_STATE_PT_LINES
            | BGFX_STATE_MSAA;
        submitMeshPass(
            segmentBorderIbh_,
            segmentBorderIndexCount_,
            borderColorUniform,
            borderMaterialUniform,
            borderState
        );
    }

    if (bgfx::isValid(seedTriangleIbh_))
    {
        const float seedColorUniform[4] = {1.0f, 0.12f, 0.12f, 1.0f};

        const uint64_t seedState = BGFX_STATE_WRITE_RGB
            | BGFX_STATE_WRITE_A
            | BGFX_STATE_DEPTH_TEST_LEQUAL
            | BGFX_STATE_MSAA;
        submitMeshPass(
            seedTriangleIbh_,
            static_cast<uint32_t>(seedTriangleIndices_.size()),
            seedColorUniform,
            materialUniform,
            seedState
        );
    }
}

void Renderer::frame()
{
    if (initialized_)
    {
        bgfx::frame();
    }
}

const char* Renderer::rendererName() const
{
    if (!initialized_)
    {
        return "Unavailable";
    }

    return bgfx::getRendererName(bgfx::getRendererType());
}

void Renderer::destroyGpuMesh()
{
    destroyGpuMeshHandles(gpuMesh_);
}

void Renderer::shutdown()
{
    clearTriangleGroups();
    clearSeedTriangle();
    destroyGpuMesh();

    destroyUniform(uBaseColor_);
    destroyUniform(uLightDir_);
    destroyUniform(uLightColor_);
    destroyUniform(uCameraPos_);
    destroyUniform(uMaterial_);

    if (bgfx::isValid(meshProgram_))
    {
        bgfx::destroy(meshProgram_);
    }
    meshProgram_ = BGFX_INVALID_HANDLE;

    if (initialized_)
    {
        bgfx::shutdown();
        initialized_ = false;
    }
}
}
