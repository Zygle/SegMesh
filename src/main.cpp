#include "obj_loader.h"
#include "renderer.h"
#include "segmentation.h"
#include "ui.h"

#include <GLFW/glfw3.h>
#include <bx/math.h>

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace
{
constexpr float kPi = 3.14159265358979323846f;
constexpr float kOrbitSensitivity = 0.004f;
constexpr float kZoomSensitivity = 0.2f;
constexpr float kMinCameraDistance = 1.0f;
constexpr float kMaxCameraDistance = 10.0f;
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

segmesh::Float3 toFloat3(const bx::Vec3& v)
{
    return {v.x, v.y, v.z};
}

float wrapAngle(float angle)
{
    return std::remainder(angle, 2.0f * kPi);
}

float clampOrbitPitch(float angle)
{
    // Stop just short of the pole so the orbit basis never flips and mirrors.
    const float limit = 0.5f * kPi - 0.01f;
    if (angle > limit)
    {
        return limit;
    }

    if (angle < -limit)
    {
        return -limit;
    }

    return angle;
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
    double mouseFramebufferX,
    double mouseFramebufferY,
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
    const segmesh::Float3 eyePosition = cameraEyePosition(uiState);
    const bx::Vec3 eye{eyePosition.x, eyePosition.y, eyePosition.z};
    const bx::Vec3 at{0.0f, 0.0f, 0.0f};

    float view[16];
    bx::mtxLookAt(view, eye, at);

    const bgfx::Caps* caps = bgfx::getCaps();
    if (caps == nullptr)
    {
        return std::nullopt;
    }

    float proj[16];
    bx::mtxProj(proj, 60.0f, aspect, 0.01f, 100.0f, caps->homogeneousDepth);

    float invView[16];
    float invProj[16];
    float model[16];
    float invModel[16];
    bx::mtxInverse(invView, view);
    bx::mtxInverse(invProj, proj);
    bx::mtxRotateY(model, modelRotation);
    bx::mtxInverse(invModel, model);

    const float ndcX = static_cast<float>(((mouseFramebufferX + 0.5) / static_cast<double>(width)) * 2.0 - 1.0);
    const float ndcY = static_cast<float>(1.0 - ((mouseFramebufferY + 0.5) / static_cast<double>(height)) * 2.0);
    const float nearClipZ = caps->homogeneousDepth ? 0.0f : -1.0f;

    const bx::Vec3 nearClip{ndcX, ndcY, nearClipZ};
    const bx::Vec3 farClip{ndcX, ndcY, 1.0f};
    const bx::Vec3 nearWorld = bx::mulH(bx::mulH(nearClip, invProj), invView);
    const bx::Vec3 farWorld = bx::mulH(bx::mulH(farClip, invProj), invView);

    const segmesh::Float3 rayOrigin = toFloat3(bx::mulH(nearWorld, invModel));
    const segmesh::Float3 rayEnd = toFloat3(bx::mulH(farWorld, invModel));
    const segmesh::Float3 rayDirection = normalize(subtract(rayEnd, rayOrigin), {0.0f, 0.0f, -1.0f});

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

void scaleCursorToFramebuffer(
    GLFWwindow* window,
    uint32_t framebufferWidth,
    uint32_t framebufferHeight,
    double mouseX,
    double mouseY,
    double& outMouseFramebufferX,
    double& outMouseFramebufferY
)
{
    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    if (windowWidth <= 0 || windowHeight <= 0)
    {
        outMouseFramebufferX = mouseX;
        outMouseFramebufferY = mouseY;
        return;
    }

    const double scaleX = static_cast<double>(framebufferWidth) / static_cast<double>(windowWidth);
    const double scaleY = static_cast<double>(framebufferHeight) / static_cast<double>(windowHeight);
    outMouseFramebufferX = mouseX * scaleX;
    outMouseFramebufferY = mouseY * scaleY;
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

const std::vector<uint32_t>& activeSeedTriangles(
    const std::vector<uint32_t>& manualSeedTriangles,
    const std::vector<uint32_t>& automaticSeedTriangles,
    bool automaticSegmentation
)
{
    return automaticSegmentation ? automaticSeedTriangles : manualSeedTriangles;
}

struct RandomWalkLabelCache
{
    bool valid = false;
    uint64_t meshGeneration = 0;
    segmesh::SegmentationModelType segmentationModelType = segmesh::SegmentationModelType::Graphical;
    std::vector<uint32_t> seedTriangles;
    std::vector<uint32_t> unmergedTriangleLabels;

    void invalidate()
    {
        valid = false;
        seedTriangles.clear();
        unmergedTriangleLabels.clear();
    }

    bool matches(
        uint64_t requestedMeshGeneration,
        segmesh::SegmentationModelType requestedSegmentationModelType,
        const std::vector<uint32_t>& requestedSeedTriangles,
        uint32_t faceCount
    ) const
    {
        return valid
            && meshGeneration == requestedMeshGeneration
            && segmentationModelType == requestedSegmentationModelType
            && seedTriangles == requestedSeedTriangles
            && unmergedTriangleLabels.size() == faceCount;
    }
};

bool syncDisplayedSeedTriangles(
    segmesh::Renderer& renderer,
    const segmesh::CpuMesh& mesh,
    const std::vector<uint32_t>& seedTriangles,
    std::string& error
)
{
    renderer.clearSeedTriangle();
    for (const uint32_t triangleIndex : seedTriangles)
    {
        if (!renderer.setSeedTriangle(mesh, triangleIndex, error))
        {
            renderer.clearSeedTriangle();
            return false;
        }
    }

    error.clear();
    return true;
}

bool buildAutomaticSeedPlan(
    const segmesh::CpuMesh& mesh,
    const segmesh::RendererUiState& uiState,
    std::vector<uint32_t>& automaticSeedPlanTriangles,
    std::string& error
)
{
    const int minSeedCount =
        uiState.automaticSegmentationMode == segmesh::AutomaticSegmentationMode::Fine ? 20 : 1;
    const uint32_t targetSeedCount = static_cast<uint32_t>(std::max(uiState.automaticSeedCount, minSeedCount));

    bool success = false;
    if (uiState.automaticSegmentationMode == segmesh::AutomaticSegmentationMode::Coarse)
    {
        success = segmesh::selectAutomaticSeedsCoarse(
            mesh,
            uiState.segmentationModelType,
            targetSeedCount,
            automaticSeedPlanTriangles,
            error
        );
    }
    else
    {
        success = segmesh::selectAutomaticSeedsFine(
            mesh,
            uiState.segmentationModelType,
            targetSeedCount,
            automaticSeedPlanTriangles,
            error
        );
    }

    if (!success)
    {
        automaticSeedPlanTriangles.clear();
        return false;
    }

    return true;
}

uint32_t automaticSeedPrefixCount(
    segmesh::RendererUiState& uiState,
    const std::vector<uint32_t>& automaticSeedPlanTriangles
)
{
    const uint32_t planSeedCount = static_cast<uint32_t>(automaticSeedPlanTriangles.size());
    if (planSeedCount == 0)
    {
        uiState.automaticStepSeedCount = 0;
        return 0;
    }

    if (!uiState.stepAutomaticSegmentation)
    {
        uiState.automaticStepSeedCount = static_cast<int>(planSeedCount);
        return planSeedCount;
    }

    const uint32_t requestedSeedCount = static_cast<uint32_t>(std::max(uiState.automaticStepSeedCount, 1));
    const uint32_t activeSeedCount = std::min(requestedSeedCount, planSeedCount);
    uiState.automaticStepSeedCount = static_cast<int>(activeSeedCount);
    return activeSeedCount;
}

bool syncAutomaticSeedPrefix(
    segmesh::Renderer& renderer,
    const segmesh::CpuMesh& mesh,
    segmesh::RendererUiState& uiState,
    const std::vector<uint32_t>& automaticSeedPlanTriangles,
    std::vector<uint32_t>& automaticSeedTriangles,
    std::string& error
)
{
    const uint32_t activeSeedCount = automaticSeedPrefixCount(uiState, automaticSeedPlanTriangles);
    if (activeSeedCount == 0)
    {
        automaticSeedTriangles.clear();
        renderer.clearSeedTriangle();
        renderer.clearTriangleGroups();
        error = "Automatic segmentation has no generated seed plan.";
        return false;
    }

    automaticSeedTriangles.assign(
        automaticSeedPlanTriangles.begin(),
        automaticSeedPlanTriangles.begin() + static_cast<std::ptrdiff_t>(activeSeedCount)
    );

    if (!syncDisplayedSeedTriangles(renderer, mesh, automaticSeedTriangles, error))
    {
        automaticSeedTriangles.clear();
        renderer.clearSeedTriangle();
        renderer.clearTriangleGroups();
        return false;
    }

    return true;
}

bool rebuildAutomaticSeeds(
    segmesh::Renderer& renderer,
    const segmesh::CpuMesh& mesh,
    segmesh::RendererUiState& uiState,
    std::vector<uint32_t>& automaticSeedPlanTriangles,
    std::vector<uint32_t>& automaticSeedTriangles,
    std::string& error
)
{
    if (!buildAutomaticSeedPlan(mesh, uiState, automaticSeedPlanTriangles, error))
    {
        automaticSeedTriangles.clear();
        renderer.clearSeedTriangle();
        renderer.clearTriangleGroups();
        return false;
    }

    return syncAutomaticSeedPrefix(
        renderer,
        mesh,
        uiState,
        automaticSeedPlanTriangles,
        automaticSeedTriangles,
        error
    );
}

bool refreshSegmentationPreview(
    segmesh::Renderer& renderer,
    const segmesh::CpuMesh& mesh,
    const std::vector<uint32_t>& seedTriangles,
    const segmesh::RendererUiState& uiState,
    uint64_t meshGeneration,
    RandomWalkLabelCache& randomWalkLabelCache,
    std::string& error
)
{
    if (!uiState.showSegmentation || seedTriangles.empty())
    {
        renderer.clearTriangleGroups();
        return true;
    }

    std::vector<uint32_t> triangleLabels;
    const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (randomWalkLabelCache.matches(meshGeneration, uiState.segmentationModelType, seedTriangles, faceCount))
    {
        triangleLabels = randomWalkLabelCache.unmergedTriangleLabels;
    }
    else
    {
        if (!segmesh::segmentMeshRandomWalk(
                mesh,
                uiState.segmentationModelType,
                seedTriangles,
                triangleLabels,
                error
            ))
        {
            randomWalkLabelCache.invalidate();
            return false;
        }

        randomWalkLabelCache.valid = true;
        randomWalkLabelCache.meshGeneration = meshGeneration;
        randomWalkLabelCache.segmentationModelType = uiState.segmentationModelType;
        randomWalkLabelCache.seedTriangles = seedTriangles;
        randomWalkLabelCache.unmergedTriangleLabels = triangleLabels;
    }

    const bool automaticStepInProgress =
        uiState.automaticSegmentation
        && uiState.stepAutomaticSegmentation
        && static_cast<int>(seedTriangles.size()) < std::max(uiState.automaticSeedCount, 1);

    if (uiState.automaticSegmentation && uiState.automaticSegmentationMode == segmesh::AutomaticSegmentationMode::Fine
        && uiState.mergeFineSegments && !automaticStepInProgress)
    {
        const uint32_t mergeTargetSegmentCount = static_cast<uint32_t>(std::max(uiState.mergeTargetSegmentCount, 1));
        if (!segmesh::mergeSegmentsByBoundaryCost(
                mesh,
                uiState.segmentationModelType,
                mergeTargetSegmentCount,
                static_cast<double>(uiState.mergeCostThreshold),
                triangleLabels,
                error
            ))
        {
            return false;
        }
    }

    if (uiState.automaticSegmentation && uiState.automaticSegmentationMode == segmesh::AutomaticSegmentationMode::Fine
        && uiState.cleanupSmallFineSegments && !automaticStepInProgress)
    {
        const uint32_t minTriangleCount = static_cast<uint32_t>(std::max(uiState.minFineSegmentTriangles, 2));
        if (!segmesh::mergeSmallSegmentsByTriangleCount(
                mesh,
                uiState.segmentationModelType,
                minTriangleCount,
                triangleLabels,
                error
            ))
        {
            return false;
        }
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
    uint64_t meshGeneration,
    RandomWalkLabelCache& randomWalkLabelCache,
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
    return refreshSegmentationPreview(
        renderer,
        mesh,
        selectedSeedTriangles,
        uiState,
        meshGeneration,
        randomWalkLabelCache,
        error
    );
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
    std::vector<uint32_t> manualSeedTriangles;
    std::vector<uint32_t> automaticSeedPlanTriangles;
    std::vector<uint32_t> automaticSeedTriangles;
    uint64_t meshGeneration = 1;
    RandomWalkLabelCache randomWalkLabelCache;
    segmesh::RendererUiState uiState{};
    std::mt19937 randomEngine(std::random_device{}());
    double previousMouseX = 0.0;
    double previousMouseY = 0.0;
    glfwGetCursorPos(window, &previousMouseX, &previousMouseY);
    bool previousRightPressed = false;
    bool orbitDragging = false;

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
        const bool previousAutomaticSegmentation = uiState.automaticSegmentation;
        const segmesh::SegmentationModelType previousSegmentationModelType = uiState.segmentationModelType;
        const segmesh::AutomaticSegmentationMode previousAutomaticSegmentationMode =
            uiState.automaticSegmentationMode;
        const int previousAutomaticSeedCount = uiState.automaticSeedCount;
        const bool previousStepAutomaticSegmentation = uiState.stepAutomaticSegmentation;
        const int previousAutomaticStepSeedCount = uiState.automaticStepSeedCount;
        const bool previousMergeFineSegments = uiState.mergeFineSegments;
        const int previousMergeTargetSegmentCount = uiState.mergeTargetSegmentCount;
        const float previousMergeCostThreshold = uiState.mergeCostThreshold;
        const bool previousCleanupSmallFineSegments = uiState.cleanupSmallFineSegments;
        const int previousMinFineSegmentTriangles = uiState.minFineSegmentTriangles;
        const std::vector<uint32_t>& seedsBeforeUi =
            activeSeedTriangles(manualSeedTriangles, automaticSeedTriangles, uiState.automaticSegmentation);
        const segmesh::RendererUiActions uiActions = segmesh::drawRendererPanel(
            modelPaths,
            selectedModelIndex,
            objPath,
            static_cast<uint32_t>(mesh.vertices.size()),
            static_cast<uint32_t>(mesh.indices.size() / 3),
            static_cast<uint32_t>(seedsBeforeUi.size()),
            renderer.rendererName(),
            uiState
        );
        if (uiState.automaticSegmentation && uiState.stepAutomaticSegmentation)
        {
            if (!previousStepAutomaticSegmentation)
            {
                uiState.automaticStepSeedCount = 1;
            }
            else if (uiActions.requestAutomaticStepReset)
            {
                uiState.automaticStepSeedCount = 1;
            }
            else if (uiActions.requestAutomaticStepFinish)
            {
                uiState.automaticStepSeedCount = automaticSeedPlanTriangles.empty()
                    ? std::numeric_limits<int>::max()
                    : static_cast<int>(automaticSeedPlanTriangles.size());
            }
            else if (uiActions.requestAutomaticStep)
            {
                const int currentStepSeedCount = std::max(uiState.automaticStepSeedCount, 1);
                uiState.automaticStepSeedCount = currentStepSeedCount < std::numeric_limits<int>::max()
                    ? currentStepSeedCount + 1
                    : currentStepSeedCount;
            }
        }
        if (uiState.automaticSegmentation)
        {
            uiState.showSegmentation = true;
        }
        const bool mouseOverUi = ImGui::MouseOverArea();
        const bool previewSettingsChanged = uiState.showSegmentation != previousShowSegmentation;
        const bool automaticSegmentationChanged =
            uiState.automaticSegmentation != previousAutomaticSegmentation;
        const bool automaticSegmentationModeChanged =
            uiState.automaticSegmentationMode != previousAutomaticSegmentationMode;
        const bool segmentationModelChanged =
            uiState.segmentationModelType != previousSegmentationModelType;
        const bool automaticSeedCountChanged = uiState.automaticSeedCount != previousAutomaticSeedCount;
        const bool automaticStepSettingsChanged =
            uiState.stepAutomaticSegmentation != previousStepAutomaticSegmentation
            || uiState.automaticStepSeedCount != previousAutomaticStepSeedCount;
        const bool mergeSettingsChanged =
            uiState.mergeFineSegments != previousMergeFineSegments
            || uiState.mergeTargetSegmentCount != previousMergeTargetSegmentCount
            || uiState.mergeCostThreshold != previousMergeCostThreshold
            || uiState.cleanupSmallFineSegments != previousCleanupSmallFineSegments
            || uiState.minFineSegmentTriangles != previousMinFineSegmentTriangles;
        const bool shouldOrbitDrag = leftPressed && !mouseOverUi;

        if (shouldOrbitDrag && !orbitDragging)
        {
            orbitDragging = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            if (glfwRawMouseMotionSupported())
            {
                glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
            }
            glfwGetCursorPos(window, &previousMouseX, &previousMouseY);
        }
        else if (!shouldOrbitDrag && orbitDragging)
        {
            orbitDragging = false;
            if (glfwRawMouseMotionSupported())
            {
                glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
            }
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            glfwGetCursorPos(window, &previousMouseX, &previousMouseY);
            mouseX = previousMouseX;
            mouseY = previousMouseY;
        }

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
                manualSeedTriangles.clear();
                automaticSeedPlanTriangles.clear();
                automaticSeedTriangles.clear();
                ++meshGeneration;
                randomWalkLabelCache.invalidate();
                meshReloaded = true;
                uiState.modelLoadError.clear();
            }
            else
            {
                uiState.modelLoadError = loadError;
            }
        }

        bool activeSeedsChanged = false;
        if (uiState.automaticSegmentation
            && (meshReloaded || automaticSegmentationChanged || automaticSegmentationModeChanged
                || segmentationModelChanged
                || automaticSeedCountChanged))
        {
            std::string autoSeedError;
            if (rebuildAutomaticSeeds(
                    renderer,
                    mesh,
                    uiState,
                    automaticSeedPlanTriangles,
                    automaticSeedTriangles,
                    autoSeedError
                ))
            {
                activeSeedsChanged = true;
                uiState.modelLoadError.clear();
            }
            else
            {
                uiState.modelLoadError = autoSeedError;
            }
        }
        else if (uiState.automaticSegmentation && automaticStepSettingsChanged)
        {
            std::string stepError;
            if (syncAutomaticSeedPrefix(
                    renderer,
                    mesh,
                    uiState,
                    automaticSeedPlanTriangles,
                    automaticSeedTriangles,
                    stepError
                ))
            {
                activeSeedsChanged = true;
                uiState.modelLoadError.clear();
            }
            else
            {
                uiState.modelLoadError = stepError;
            }
        }
        else if (automaticSegmentationChanged)
        {
            std::string seedDisplayError;
            const std::vector<uint32_t>& currentSeeds =
                activeSeedTriangles(manualSeedTriangles, automaticSeedTriangles, uiState.automaticSegmentation);
            if (syncDisplayedSeedTriangles(renderer, mesh, currentSeeds, seedDisplayError))
            {
                activeSeedsChanged = true;
                uiState.modelLoadError.clear();
            }
            else
            {
                uiState.modelLoadError = seedDisplayError;
            }
        }

        if (orbitDragging)
        {
            const double mouseDeltaX = mouseX - previousMouseX;
            const double mouseDeltaY = mouseY - previousMouseY;
            uiState.cameraYaw = wrapAngle(uiState.cameraYaw + static_cast<float>(mouseDeltaX) * kOrbitSensitivity);
            uiState.cameraPitch =
                clampOrbitPitch(uiState.cameraPitch + static_cast<float>(mouseDeltaY) * kOrbitSensitivity);
        }

        if (!mouseOverUi)
        {

            if (scroll != 0)
            {
                uiState.cameraDistance = std::clamp(
                    uiState.cameraDistance - static_cast<float>(scroll) * kZoomSensitivity,
                    kMinCameraDistance,
                    kMaxCameraDistance
                );
            }

            if (!uiState.automaticSegmentation && rightPressed && !previousRightPressed)
            {
                double mouseFramebufferX = mouseX;
                double mouseFramebufferY = mouseY;
                scaleCursorToFramebuffer(
                    window,
                    static_cast<uint32_t>(width),
                    static_cast<uint32_t>(height),
                    mouseX,
                    mouseY,
                    mouseFramebufferX,
                    mouseFramebufferY
                );

                const auto pickedTriangle = pickTriangleUnderCursor(
                    mesh,
                    mouseFramebufferX,
                    mouseFramebufferY,
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
                            manualSeedTriangles,
                            uiState,
                            meshGeneration,
                            randomWalkLabelCache,
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

        if (previewSettingsChanged || meshReloaded || activeSeedsChanged || mergeSettingsChanged
            || segmentationModelChanged)
        {
            std::string previewError;
            const std::vector<uint32_t>& currentSeeds =
                activeSeedTriangles(manualSeedTriangles, automaticSeedTriangles, uiState.automaticSegmentation);
            if (!refreshSegmentationPreview(
                    renderer,
                    mesh,
                    currentSeeds,
                    uiState,
                    meshGeneration,
                    randomWalkLabelCache,
                    previewError
                ))
            {
                uiState.modelLoadError = previewError;
            }
        }

        if (uiActions.requestClearSeed)
        {
            renderer.clearSeedTriangle();
            manualSeedTriangles.clear();
            uiState.modelLoadError.clear();

            std::string previewError;
            if (!refreshSegmentationPreview(
                    renderer,
                    mesh,
                    manualSeedTriangles,
                    uiState,
                    meshGeneration,
                    randomWalkLabelCache,
                    previewError
                ))
            {
                uiState.modelLoadError = previewError;
            }
        }

        if (uiActions.requestRandomSeed)
        {
            const auto randomSeed = pickRandomSeedTriangle(mesh, manualSeedTriangles, randomEngine);
            if (!randomSeed.has_value())
            {
                uiState.modelLoadError = "No unseeded triangles left. Use Clear seeds to reset.";
            }
            else
            {
                std::string seedError;
                if (addSeedTriangle(
                        renderer,
                        mesh,
                        *randomSeed,
                        manualSeedTriangles,
                        uiState,
                        meshGeneration,
                        randomWalkLabelCache,
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

        previousMouseX = mouseX;
        previousMouseY = mouseY;
        previousRightPressed = rightPressed;

        renderer.renderScene(static_cast<uint32_t>(width), static_cast<uint32_t>(height), modelRotation, uiState);
        imguiEndFrame();
        renderer.frame();

        if (uiState.autoRotate)
        {
            modelRotation += uiState.rotateSpeed * dt;
        }
    }

    imguiDestroy();
    if (glfwRawMouseMotionSupported())
    {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
    }
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    renderer.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
