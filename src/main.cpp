#include "obj_loader.h"
#include "renderer.h"
#include "segmentation.h"
#include "ui.h"

#include <GLFW/glfw3.h>

#include "imgui.h"

#include <algorithm>
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
constexpr float kPi = 3.14159265358979323846f;
constexpr float kOrbitSensitivity = 0.01f;
constexpr float kZoomSensitivity = 0.2f;
constexpr float kMinCameraDistance = 1.0f;
constexpr float kMaxCameraDistance = 10.0f;
constexpr float kMinCameraPitch = -1.45f;
constexpr float kMaxCameraPitch = 1.45f;
constexpr float kPickEpsilon = 1.0e-8f;

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

segmesh::Float3 add(const segmesh::Float3& a, const segmesh::Float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

segmesh::Float3 subtract(const segmesh::Float3& a, const segmesh::Float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

segmesh::Float3 scale(const segmesh::Float3& v, float scalar)
{
    return {v.x * scalar, v.y * scalar, v.z * scalar};
}

float dot(const segmesh::Float3& a, const segmesh::Float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

segmesh::Float3 cross(const segmesh::Float3& a, const segmesh::Float3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float length(const segmesh::Float3& v)
{
    return std::sqrt(dot(v, v));
}

segmesh::Float3 normalize(const segmesh::Float3& v, const segmesh::Float3& fallback)
{
    const float len = length(v);
    if (len < kPickEpsilon)
    {
        return fallback;
    }

    return scale(v, 1.0f / len);
}

segmesh::Float3 rotateY(const segmesh::Float3& v, float angle)
{
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {
        c * v.x + s * v.z,
        v.y,
        -s * v.x + c * v.z,
    };
}

segmesh::Float3 cameraEyePosition(const segmesh::RendererUiState& uiState)
{
    const float cosPitch = std::cos(uiState.cameraPitch);
    return {
        std::sin(uiState.cameraYaw) * cosPitch * uiState.cameraDistance,
        std::sin(uiState.cameraPitch) * uiState.cameraDistance,
        std::cos(uiState.cameraYaw) * cosPitch * uiState.cameraDistance,
    };
}

bool rayTriangleIntersection(
    const segmesh::Float3& rayOrigin,
    const segmesh::Float3& rayDirection,
    const segmesh::Float3& p0,
    const segmesh::Float3& p1,
    const segmesh::Float3& p2,
    float& outDistance
)
{
    const segmesh::Float3 edge01 = subtract(p1, p0);
    const segmesh::Float3 edge02 = subtract(p2, p0);
    const segmesh::Float3 p = cross(rayDirection, edge02);
    const float determinant = dot(edge01, p);
    if (std::abs(determinant) < kPickEpsilon)
    {
        return false;
    }

    const float inverseDeterminant = 1.0f / determinant;
    const segmesh::Float3 t = subtract(rayOrigin, p0);
    const float u = dot(t, p) * inverseDeterminant;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }

    const segmesh::Float3 q = cross(t, edge01);
    const float v = dot(rayDirection, q) * inverseDeterminant;
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }

    const float distance = dot(edge02, q) * inverseDeterminant;
    if (distance <= kPickEpsilon)
    {
        return false;
    }

    outDistance = distance;
    return true;
}

std::optional<uint32_t> pickTriangleUnderCursor(
    const segmesh::CpuMesh& mesh,
    double mouseX,
    double mouseY,
    uint32_t width,
    uint32_t height,
    float modelRotation,
    const segmesh::RendererUiState& uiState
)
{
    if (width == 0 || height == 0)
    {
        return std::nullopt;
    }

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float tanHalfFov = std::tan(60.0f * kPi / 360.0f);

    const segmesh::Float3 eye = cameraEyePosition(uiState);
    const segmesh::Float3 forward = normalize(scale(eye, -1.0f), {0.0f, 0.0f, -1.0f});
    const segmesh::Float3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}), {1.0f, 0.0f, 0.0f});
    const segmesh::Float3 up = normalize(cross(right, forward), {0.0f, 1.0f, 0.0f});

    const float ndcX = static_cast<float>(((mouseX + 0.5) / static_cast<double>(width)) * 2.0 - 1.0);
    const float ndcY = static_cast<float>(1.0 - ((mouseY + 0.5) / static_cast<double>(height)) * 2.0);

    const segmesh::Float3 rayDirectionWorld = normalize(
        add(
            add(forward, scale(right, ndcX * aspect * tanHalfFov)),
            scale(up, ndcY * tanHalfFov)
        ),
        forward
    );

    const segmesh::Float3 rayOrigin = rotateY(eye, -modelRotation);
    const segmesh::Float3 rayDirection = normalize(rotateY(rayDirectionWorld, -modelRotation), rayDirectionWorld);

    float closestDistance = std::numeric_limits<float>::max();
    std::optional<uint32_t> closestTriangle;
    const uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    for (uint32_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex)
    {
        const uint32_t indexOffset = triangleIndex * 3;
        const segmesh::MeshVertex& v0 = mesh.vertices[mesh.indices[indexOffset + 0]];
        const segmesh::MeshVertex& v1 = mesh.vertices[mesh.indices[indexOffset + 1]];
        const segmesh::MeshVertex& v2 = mesh.vertices[mesh.indices[indexOffset + 2]];

        float hitDistance = 0.0f;
        if (!rayTriangleIntersection(
                rayOrigin,
                rayDirection,
                {v0.x, v0.y, v0.z},
                {v1.x, v1.y, v1.z},
                {v2.x, v2.y, v2.z},
                hitDistance
            ))
        {
            continue;
        }

        if (hitDistance < closestDistance)
        {
            closestDistance = hitDistance;
            closestTriangle = triangleIndex;
        }
    }

    return closestTriangle;
}

bool triangleAlreadySeeded(const std::vector<uint32_t>& seededTriangles, uint32_t triangleIndex)
{
    return std::find(seededTriangles.begin(), seededTriangles.end(), triangleIndex) != seededTriangles.end();
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

bool refreshSegmentationPreview(
    segmesh::Renderer& renderer,
    const segmesh::CpuMesh& mesh,
    const std::vector<uint32_t>& seedTriangles,
    const segmesh::RendererUiState& uiState,
    std::string& error
)
{
    if (!uiState.showSegmentation || seedTriangles.empty())
    {
        renderer.clearTriangleGroups();
        return true;
    }

    std::vector<uint32_t> triangleLabels;
    if (!segmesh::segmentMeshRandomWalk(mesh, seedTriangles, triangleLabels, error))
    {
        return false;
    }

    uint32_t segmentCount = 0;
    for (const uint32_t label : triangleLabels)
    {
        segmentCount = std::max(segmentCount, label + 1);
    }

    if (segmentCount == 0)
    {
        renderer.clearTriangleGroups();
        return true;
    }

    return renderer.setTriangleGroups(mesh, triangleLabels, segmentCount, error);
}

bool addSeedTriangle(
    segmesh::Renderer& renderer,
    const segmesh::CpuMesh& mesh,
    uint32_t triangleIndex,
    std::vector<uint32_t>& selectedSeedTriangles,
    const segmesh::RendererUiState& uiState,
    std::string& error
)
{
    if (triangleAlreadySeeded(selectedSeedTriangles, triangleIndex))
    {
        error.clear();
        return true;
    }

    std::string seedError;
    if (!renderer.setSeedTriangle(mesh, triangleIndex, seedError))
    {
        error = seedError;
        return false;
    }

    selectedSeedTriangles.push_back(triangleIndex);
    return refreshSegmentationPreview(renderer, mesh, selectedSeedTriangles, uiState, error);
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
    double previousMouseX = 0.0;
    double previousMouseY = 0.0;
    glfwGetCursorPos(window, &previousMouseX, &previousMouseY);
    bool previousLeftPressed = false;
    bool previousRightPressed = false;

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

        double mouseX = 0.0;
        double mouseY = 0.0;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        uint8_t mouseButtons = 0;
        const bool leftPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        const bool rightPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        if (leftPressed)
        {
            mouseButtons |= IMGUI_MBUT_LEFT;
        }
        if (rightPressed)
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

        const bool previousShowSegmentation = uiState.showSegmentation;
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
        const bool mouseOverUi = ImGui::MouseOverArea();
        const bool previewSettingsChanged = uiState.showSegmentation != previousShowSegmentation;

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

        if (!mouseOverUi)
        {
            const double mouseDeltaX = mouseX - previousMouseX;
            const double mouseDeltaY = mouseY - previousMouseY;
            if (leftPressed && previousLeftPressed)
            {
                uiState.cameraYaw -= static_cast<float>(mouseDeltaX) * kOrbitSensitivity;
                uiState.cameraPitch = std::clamp(
                    uiState.cameraPitch - static_cast<float>(mouseDeltaY) * kOrbitSensitivity,
                    kMinCameraPitch,
                    kMaxCameraPitch
                );
            }

            if (scroll != 0)
            {
                uiState.cameraDistance = std::clamp(
                    uiState.cameraDistance - static_cast<float>(scroll) * kZoomSensitivity,
                    kMinCameraDistance,
                    kMaxCameraDistance
                );
            }

            if (rightPressed && !previousRightPressed)
            {
                const auto pickedTriangle = pickTriangleUnderCursor(
                    mesh,
                    mouseX,
                    mouseY,
                    static_cast<uint32_t>(width),
                    static_cast<uint32_t>(height),
                    modelRotation,
                    uiState
                );
                if (!pickedTriangle.has_value())
                {
                    uiState.modelLoadError = "No triangle found under the cursor.";
                }
                else
                {
                    std::string seedError;
                    if (addSeedTriangle(
                            renderer,
                            mesh,
                            *pickedTriangle,
                            selectedSeedTriangles,
                            uiState,
                            seedError
                        ))
                    {
                        uiState.modelLoadError.clear();
                    }
                    else
                    {
                        uiState.modelLoadError = seedError;
                    }
                }
            }
        }

        if (previewSettingsChanged || meshReloaded)
        {
            std::string previewError;
            if (!refreshSegmentationPreview(renderer, mesh, selectedSeedTriangles, uiState, previewError))
            {
                uiState.modelLoadError = previewError;
            }
        }

        if (uiActions.requestClearSeed)
        {
            renderer.clearSeedTriangle();
            selectedSeedTriangles.clear();
            uiState.modelLoadError.clear();

            std::string previewError;
            if (!refreshSegmentationPreview(renderer, mesh, selectedSeedTriangles, uiState, previewError))
            {
                uiState.modelLoadError = previewError;
            }
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
                if (addSeedTriangle(renderer, mesh, *randomSeed, selectedSeedTriangles, uiState, seedError))
                {
                    uiState.modelLoadError.clear();
                }
                else
                {
                    uiState.modelLoadError = seedError;
                }
            }
        }

        previousMouseX = mouseX;
        previousMouseY = mouseY;
        previousLeftPressed = leftPressed;
        previousRightPressed = rightPressed;

        renderer.renderScene(static_cast<uint32_t>(width), static_cast<uint32_t>(height), modelRotation, uiState);
        imguiEndFrame();
        renderer.frame();
    }

    imguiDestroy();
    renderer.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
