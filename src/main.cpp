#include "obj_loader.h"
#include "renderer.h"
#include "ui.h"

#include <GLFW/glfw3.h>

#include "imgui.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace
{
double g_scrollDelta = 0.0;
int g_inputChar = -1;

void onScroll(GLFWwindow*, double, double yoffset)
{
    g_scrollDelta += yoffset;
}

void onChar(GLFWwindow*, unsigned int codepoint)
{
    g_inputChar = static_cast<int>(codepoint);
}

std::optional<uint32_t> pickRandomSeedTriangle(
    const segmesh::CpuMesh& mesh,
    const std::vector<uint32_t>& seededTriangles,
    std::mt19937& randomEngine
)
{
    const uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (triangleCount == 0)
    {
        return std::nullopt;
    }

    std::vector<bool> isSeeded(triangleCount, false);
    uint32_t seededCount = 0;
    for (const uint32_t triangleIndex : seededTriangles)
    {
        if (triangleIndex < triangleCount && !isSeeded[triangleIndex])
        {
            isSeeded[triangleIndex] = true;
            ++seededCount;
        }
    }

    if (seededCount >= triangleCount)
    {
        return std::nullopt;
    }

    if (mesh.faceAreas.size() == triangleCount)
    {
        std::vector<double> weights;
        weights.reserve(mesh.faceAreas.size());

        double areaSum = 0.0;
        for (uint32_t i = 0; i < triangleCount; ++i)
        {
            const float area = mesh.faceAreas[static_cast<std::size_t>(i)];
            if (isSeeded[i])
            {
                weights.push_back(0.0);
                continue;
            }

            const double weight = (std::isfinite(area) && area > 0.0f) ? static_cast<double>(area) : 0.0;
            weights.push_back(weight);
            areaSum += weight;
        }

        if (areaSum > 0.0)
        {
            std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
            return static_cast<uint32_t>(dist(randomEngine));
        }
    }

    std::uniform_int_distribution<uint32_t> dist(0, triangleCount - 1);
    uint32_t candidate = dist(randomEngine);
    while (isSeeded[candidate])
    {
        candidate = dist(randomEngine);
    }
    return candidate;
}

std::vector<uint32_t> buildTriangleGroups(const segmesh::CpuMesh& mesh, uint32_t groupCount)
{
    const uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    std::vector<uint32_t> triangleGroups(triangleCount, 0);
    if (triangleCount == 0 || groupCount == 0)
    {
        return triangleGroups;
    }

    if (mesh.faceCentroids.size() != triangleCount)
    {
        for (uint32_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex)
        {
            triangleGroups[triangleIndex] = triangleIndex % groupCount;
        }
        return triangleGroups;
    }

    segmesh::Float3 center{0.0f, 0.0f, 0.0f};
    for (const segmesh::Float3& centroid : mesh.faceCentroids)
    {
        center.x += centroid.x;
        center.y += centroid.y;
        center.z += centroid.z;
    }

    const float inverseTriangleCount = 1.0f / static_cast<float>(triangleCount);
    center.x *= inverseTriangleCount;
    center.y *= inverseTriangleCount;
    center.z *= inverseTriangleCount;

    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kTwoPi = 6.28318530717958647692f;

    for (uint32_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex)
    {
        const segmesh::Float3& centroid = mesh.faceCentroids[triangleIndex];
        const float angle = std::atan2(centroid.z - center.z, centroid.x - center.x);
        float normalizedAngle = (angle + kPi) / kTwoPi;
        if (normalizedAngle >= 1.0f)
        {
            normalizedAngle = std::nextafter(1.0f, 0.0f);
        }

        uint32_t groupIndex = static_cast<uint32_t>(normalizedAngle * static_cast<float>(groupCount));
        if (groupIndex >= groupCount)
        {
            groupIndex = groupCount - 1;
        }
        triangleGroups[triangleIndex] = groupIndex;
    }

    return triangleGroups;
}

bool refreshTriangleGroups(
    segmesh::Renderer& renderer,
    const segmesh::CpuMesh& mesh,
    const segmesh::RendererUiState& uiState,
    std::string& error
)
{
    if (!uiState.showGroups)
    {
        renderer.clearTriangleGroups();
        return true;
    }

    const uint32_t groupCount = static_cast<uint32_t>(uiState.groupCount);
    return renderer.setTriangleGroups(mesh, buildTriangleGroups(mesh, groupCount), groupCount, error);
}
}

int main(int argc, char** argv)
{
    std::vector<std::filesystem::path> modelPaths;
    const std::vector<std::filesystem::path> modelSearchDirs = {
        "assets",
        "../assets",
        "../../assets",
    };

    const auto modelDir = segmesh::findFirstExisting({
        modelSearchDirs[0],
        modelSearchDirs[1],
        modelSearchDirs[2],
    });
    if (modelDir.has_value())
    {
        modelPaths = segmesh::findObjFiles(*modelDir);
    }

    int selectedModelIndex = 0;
    if (argc > 1)
    {
        const std::filesystem::path cliPath(argv[1]);
        if (!std::filesystem::exists(cliPath))
        {
            std::cerr << "OBJ file not found: " << cliPath << "\n";
            return 1;
        }

        const int existingIndex = segmesh::findModelIndex(modelPaths, cliPath);
        if (existingIndex >= 0)
        {
            selectedModelIndex = existingIndex;
        }
        else
        {
            modelPaths.insert(modelPaths.begin(), cliPath);
            selectedModelIndex = 0;
        }
    }

    if (modelPaths.empty())
    {
        std::vector<std::filesystem::path> defaultObjCandidates;
        defaultObjCandidates.reserve(modelSearchDirs.size());
        for (const std::filesystem::path& dir : modelSearchDirs)
        {
            defaultObjCandidates.push_back(dir / "model.obj");
        }

        const auto defaultObj = segmesh::findFirstExisting(defaultObjCandidates);

        if (!defaultObj.has_value())
        {
            std::cerr << "No OBJ file found. Add .obj files to assets or pass one on the command line.\n";
            return 1;
        }

        modelPaths.push_back(*defaultObj);
        selectedModelIndex = 0;
    }

    std::filesystem::path objPath = modelPaths[static_cast<std::size_t>(selectedModelIndex)];

    segmesh::CpuMesh mesh{};
    std::string error;
    if (!segmesh::loadObjMesh(objPath, mesh, error))
    {
        std::cerr << error << "\n";
        return 1;
    }

    if (!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "SegMesh - OBJ Renderer", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }

    glfwSetScrollCallback(window, onScroll);
    glfwSetCharCallback(window, onChar);

    int width = 0;
    int height = 0;
    while (width == 0 || height == 0)
    {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &width, &height);
    }

    segmesh::Renderer renderer;
    if (!renderer.initialize(window, static_cast<uint32_t>(width), static_cast<uint32_t>(height), error))
    {
        std::cerr << error << "\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    if (!renderer.loadMesh(mesh, error))
    {
        std::cerr << error << "\n";
        renderer.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    imguiCreate(18.0f);

    float modelRotation = 0.0f;
    std::vector<uint32_t> selectedSeedTriangles;
    segmesh::RendererUiState uiState{};
    std::mt19937 randomEngine(std::random_device{}());

    float previousTime = static_cast<float>(glfwGetTime());
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        int newWidth = 0;
        int newHeight = 0;
        glfwGetFramebufferSize(window, &newWidth, &newHeight);
        if (newWidth != width || newHeight != height)
        {
            width = newWidth;
            height = newHeight;
            if (width > 0 && height > 0)
            {
                renderer.resize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
            }
        }

        if (width == 0 || height == 0)
        {
            continue;
        }

        const float now = static_cast<float>(glfwGetTime());
        const float dt = now - previousTime;
        previousTime = now;
        if (uiState.autoRotate)
        {
            modelRotation += uiState.rotateSpeed * dt;
        }

        renderer.renderScene(static_cast<uint32_t>(width), static_cast<uint32_t>(height), modelRotation, uiState);

        double mouseX = 0.0;
        double mouseY = 0.0;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        uint8_t mouseButtons = 0;
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            mouseButtons |= IMGUI_MBUT_LEFT;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
        {
            mouseButtons |= IMGUI_MBUT_RIGHT;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
        {
            mouseButtons |= IMGUI_MBUT_MIDDLE;
        }

        const int32_t scroll = static_cast<int32_t>(std::lround(g_scrollDelta));
        g_scrollDelta = 0.0;
        const int inputChar = g_inputChar;
        g_inputChar = -1;

        imguiBeginFrame(
            static_cast<int32_t>(mouseX),
            static_cast<int32_t>(mouseY),
            mouseButtons,
            scroll,
            static_cast<uint16_t>(width),
            static_cast<uint16_t>(height),
            inputChar,
            1
        );

        const bool previousShowGroups = uiState.showGroups;
        const int previousGroupCount = uiState.groupCount;
        const segmesh::RendererUiActions uiActions = segmesh::drawRendererPanel(
            modelPaths,
            selectedModelIndex,
            objPath,
            static_cast<uint32_t>(mesh.vertices.size()),
            static_cast<uint32_t>(mesh.indices.size() / 3),
            static_cast<uint32_t>(selectedSeedTriangles.size()),
            renderer.rendererName(),
            uiState
        );
        const bool groupSettingsChanged = uiState.showGroups != previousShowGroups
            || uiState.groupCount != previousGroupCount;

        const int pendingModelIndex = uiActions.pendingModelIndex;
        bool meshReloaded = false;
        if (pendingModelIndex != selectedModelIndex)
        {
            segmesh::CpuMesh newMesh{};
            std::string loadError;
            const std::filesystem::path newPath = modelPaths[static_cast<std::size_t>(pendingModelIndex)];
            if (segmesh::loadObjMesh(newPath, newMesh, loadError) && renderer.loadMesh(newMesh, loadError))
            {
                mesh = std::move(newMesh);
                objPath = newPath;
                selectedModelIndex = pendingModelIndex;
                modelRotation = 0.0f;
                selectedSeedTriangles.clear();
                meshReloaded = true;
                uiState.modelLoadError.clear();
            }
            else
            {
                uiState.modelLoadError = loadError;
            }
        }

        if (groupSettingsChanged || meshReloaded)
        {
            std::string groupError;
            if (!refreshTriangleGroups(renderer, mesh, uiState, groupError))
            {
                uiState.modelLoadError = groupError;
            }
        }

        if (uiActions.requestClearSeed)
        {
            renderer.clearSeedTriangle();
            selectedSeedTriangles.clear();
            uiState.modelLoadError.clear();
        }

        if (uiActions.requestRandomSeed)
        {
            const auto randomSeed = pickRandomSeedTriangle(mesh, selectedSeedTriangles, randomEngine);
            if (!randomSeed.has_value())
            {
                uiState.modelLoadError = "No unseeded triangles left. Use Clear seeds to reset.";
            }
            else
            {
                std::string seedError;
                if (renderer.setSeedTriangle(mesh, *randomSeed, seedError))
                {
                    selectedSeedTriangles.push_back(*randomSeed);
                    uiState.modelLoadError.clear();
                }
                else
                {
                    uiState.modelLoadError = seedError;
                }
            }
        }

        imguiEndFrame();
        renderer.frame();
    }

    imguiDestroy();
    renderer.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
