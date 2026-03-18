#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace segmesh
{
struct RendererUiState
{
    float cameraDistance = 3.0f;
    float cameraYaw = 0.0f;
    float cameraPitch = 0.35f;
    float rotateSpeed = 0.8f;
    bool autoRotate = true;

    float baseColor[3] = {0.78f, 0.78f, 0.78f};
    float lightDirection[3] = {-0.45f, -0.9f, -0.25f};
    float lightColor[3] = {1.0f, 0.98f, 0.95f};
    float lightIntensity = 1.5f;
    float ambientStrength = 0.22f;
    float specularStrength = 0.32f;
    float shininess = 64.0f;

    bool showSegmentation = false;

    std::string modelLoadError;
};

struct RendererUiActions
{
    int pendingModelIndex = 0;
    bool requestRandomSeed = false;
    bool requestClearSeed = false;
};

RendererUiActions drawRendererPanel(
    const std::vector<std::filesystem::path>& modelPaths,
    int selectedModelIndex,
    const std::filesystem::path& objPath,
    uint32_t vertexCount,
    uint32_t triangleCount,
    uint32_t seedTriangleCount,
    const char* rendererName,
    RendererUiState& uiState
);
}
