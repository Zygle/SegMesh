#include "border_curves.h"
#include "obj_loader.h"
#include "renderer.h"
#include "select.h"
#include "segmentation.h"
#include "segmentation_session.h"
#include "ui.h"

#include <GLFW/glfw3.h>
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

bool syncAutomaticSeedPrefix(
    segmesh::Renderer& renderer,
    const segmesh::CpuMesh& mesh,
    segmesh::RendererUiState& uiState,
    segmesh::SegmentationSession& session,
    std::string& error
)
{
    if (!session.selectAutomaticSeedPrefix(
            uiState.segmentation,
            uiState.automaticStepSeedCount,
            error
        ))
    {
        renderer.clearSeedTriangle();
        renderer.clearTriangleGroups();
        return false;
    }

    if (!syncDisplayedSeedTriangles(renderer, mesh, session.automaticSeedTriangles, error))
    {
        session.automaticSeedTriangles.clear();
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
    segmesh::SegmentationSession& session,
    std::string& error
)
{
    if (!session.buildAutomaticSeedPlan(mesh, uiState.segmentation, error))
    {
        session.automaticSeedTriangles.clear();
        renderer.clearSeedTriangle();
        renderer.clearTriangleGroups();
        return false;
    }

    return syncAutomaticSeedPrefix(
        renderer,
        mesh,
        uiState,
        session,
        error
    );
}

bool refreshSegmentationPreview(
    segmesh::Renderer& renderer,
    const segmesh::CpuMesh& mesh,
    const std::vector<uint32_t>& seedTriangles,
    const segmesh::RendererUiState& uiState,
    segmesh::SegmentationSession& session,
    std::string& error
)
{
    session.displayedTriangleLabels.clear();
    if (!uiState.rendering.showSegmentation || seedTriangles.empty())
    {
        renderer.clearTriangleGroups();
        return true;
    }

    std::vector<uint32_t> triangleLabels;
    std::vector<float> triangleConfidence;
    uint32_t segmentCount = 0;
    if (!session.buildPreviewLabels(
            mesh,
            uiState.segmentation,
            seedTriangles,
            triangleLabels,
            triangleConfidence,
            segmentCount,
            error
        ))
    {
        return false;
    }

    if (segmentCount == 0)
    {
        renderer.clearTriangleGroups();
        return true;
    }

    if (!renderer.setTriangleGroups(
            mesh,
            triangleLabels,
            triangleConfidence,
            segmentCount,
            error
        ))
    {
        return false;
    }

    segmesh::BorderCurveSettings borderCurveSettings{};
    borderCurveSettings.fitTolerance = static_cast<double>(uiState.borders.fitTolerance);
    borderCurveSettings.drawingTolerance = static_cast<double>(uiState.borders.drawingTolerance);

    segmesh::BorderCurveOverlay borderCurveOverlay;
    if (!segmesh::buildSegmentBorderOverlay(
            mesh,
            triangleLabels,
            borderCurveSettings,
            borderCurveOverlay,
            error
        ))
    {
        renderer.clearSegmentBorderOverlay();
        return false;
    }

    if (!renderer.setSegmentBorderOverlay(borderCurveOverlay, error))
    {
        renderer.clearSegmentBorderOverlay();
        return false;
    }

    session.displayedTriangleLabels = std::move(triangleLabels);
    return true;
}

bool addSeedTriangle(
    segmesh::Renderer& renderer,
    const segmesh::CpuMesh& mesh,
    uint32_t triangleIndex,
    std::vector<uint32_t>& selectedSeedTriangles,
    const segmesh::RendererUiState& uiState,
    segmesh::SegmentationSession& session,
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
        session,
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
    segmesh::SegmentationSession session;
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

        const bool previousShowSegmentation = uiState.rendering.showSegmentation;
        const bool previousAutomaticSegmentation = uiState.segmentation.automatic;
        const segmesh::SegmentationModelType previousSegmentationModelType = uiState.segmentation.model;
        const segmesh::AutomaticSegmentationMode previousAutomaticSegmentationMode =
            uiState.segmentation.automaticMode;
        const int previousAutomaticSeedCount = uiState.segmentation.automaticSeedCount;
        const bool previousStepAutomaticSegmentation = uiState.segmentation.stepByStep;
        const int previousAutomaticStepSeedCount = uiState.automaticStepSeedCount;
        const bool previousMergeFineSegments = uiState.segmentation.mergeFineSegments;
        const int previousMergeTargetSegmentCount = uiState.segmentation.mergeTargetSegmentCount;
        const bool previousForceMergeTargetSegmentCount = uiState.segmentation.forceMergeTargetSegmentCount;
        const float previousMergeCostThreshold = uiState.segmentation.mergeCostThreshold;
        const bool previousCleanupSmallFineSegments = uiState.segmentation.cleanupSmallFineSegments;
        const int previousMinFineSegmentTriangles = uiState.segmentation.minFineSegmentTriangles;
        const float previousBorderCurveFitTolerance = uiState.borders.fitTolerance;
        const float previousBorderCurveDrawingTolerance = uiState.borders.drawingTolerance;
        const bool previousManualSegmentMerge = uiState.manualSegmentMerge;
        const std::vector<uint32_t>& seedsBeforeUi =
            session.activeSeeds(uiState.segmentation.automatic);
        const bool hasSegmentationPreview = uiState.rendering.showSegmentation
            && session.displayedTriangleLabels.size() == mesh.indices.size() / 3;
        int32_t selectedMergeSegment = -1;
        if (session.selectedMergeTriangle.has_value()
            && *session.selectedMergeTriangle < session.displayedTriangleLabels.size()
            && session.displayedTriangleLabels[*session.selectedMergeTriangle]
                <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
        {
            selectedMergeSegment = static_cast<int32_t>(
                session.displayedTriangleLabels[*session.selectedMergeTriangle]
            );
        }
        const segmesh::RendererUiActions uiActions = segmesh::drawRendererPanel(
            modelPaths,
            selectedModelIndex,
            objPath,
            static_cast<uint32_t>(mesh.vertices.size()),
            static_cast<uint32_t>(mesh.indices.size() / 3),
            static_cast<uint32_t>(seedsBeforeUi.size()),
            hasSegmentationPreview,
            selectedMergeSegment,
            static_cast<uint32_t>(session.manualSegmentMerges.size()),
            renderer.rendererName(),
            uiState
        );
        if (uiState.segmentation.automatic && uiState.segmentation.stepByStep)
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
                uiState.automaticStepSeedCount = session.automaticSeedPlanTriangles.empty()
                    ? std::numeric_limits<int>::max()
                    : static_cast<int>(session.automaticSeedPlanTriangles.size());
            }
            else if (uiActions.requestAutomaticStep)
            {
                const int currentStepSeedCount = std::max(uiState.automaticStepSeedCount, 1);
                uiState.automaticStepSeedCount = currentStepSeedCount < std::numeric_limits<int>::max()
                    ? currentStepSeedCount + 1
                    : currentStepSeedCount;
            }
        }
        if (uiState.segmentation.automatic)
        {
            uiState.rendering.showSegmentation = true;
        }
        const bool mouseOverUi = ImGui::MouseOverArea();
        const bool previewSettingsChanged = uiState.rendering.showSegmentation != previousShowSegmentation;
        const bool automaticSegmentationChanged =
            uiState.segmentation.automatic != previousAutomaticSegmentation;
        const bool automaticSegmentationModeChanged =
            uiState.segmentation.automaticMode != previousAutomaticSegmentationMode;
        const bool segmentationModelChanged =
            uiState.segmentation.model != previousSegmentationModelType;
        const bool automaticSeedCountChanged = uiState.segmentation.automaticSeedCount != previousAutomaticSeedCount;
        const bool automaticStepSettingsChanged =
            uiState.segmentation.stepByStep != previousStepAutomaticSegmentation
            || uiState.automaticStepSeedCount != previousAutomaticStepSeedCount;
        const bool mergeSettingsChanged =
            uiState.segmentation.mergeFineSegments != previousMergeFineSegments
            || uiState.segmentation.mergeTargetSegmentCount != previousMergeTargetSegmentCount
            || uiState.segmentation.forceMergeTargetSegmentCount != previousForceMergeTargetSegmentCount
            || uiState.segmentation.mergeCostThreshold != previousMergeCostThreshold
            || uiState.segmentation.cleanupSmallFineSegments != previousCleanupSmallFineSegments
            || uiState.segmentation.minFineSegmentTriangles != previousMinFineSegmentTriangles;
        const bool borderCurveSettingsChanged =
            uiState.borders.fitTolerance != previousBorderCurveFitTolerance
            || uiState.borders.drawingTolerance != previousBorderCurveDrawingTolerance;
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
                session.resetForMesh();
                uiState.manualSegmentMerge = false;
                meshReloaded = true;
                uiState.modelLoadError.clear();
            }
            else
            {
                uiState.modelLoadError = loadError;
            }
        }

        bool activeSeedsChanged = false;
        if (uiState.segmentation.automatic
            && (meshReloaded || automaticSegmentationChanged || automaticSegmentationModeChanged
                || segmentationModelChanged
                || automaticSeedCountChanged))
        {
            std::string autoSeedError;
            if (rebuildAutomaticSeeds(
                    renderer,
                    mesh,
                    uiState,
                    session,
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
        else if (uiState.segmentation.automatic && automaticStepSettingsChanged)
        {
            std::string stepError;
            if (syncAutomaticSeedPrefix(
                    renderer,
                    mesh,
                    uiState,
                    session,
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
                session.activeSeeds(uiState.segmentation.automatic);
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

        const bool segmentationBasisChanged = meshReloaded
            || activeSeedsChanged
            || automaticSegmentationChanged
            || automaticSegmentationModeChanged
            || segmentationModelChanged
            || automaticSeedCountChanged
            || automaticStepSettingsChanged
            || mergeSettingsChanged;
        if (segmentationBasisChanged)
        {
            session.manualSegmentMerges.clear();
            session.selectedMergeTriangle.reset();
            renderer.clearHighlightedSegment();
        }

        if (previousManualSegmentMerge && !uiState.manualSegmentMerge)
        {
            session.selectedMergeTriangle.reset();
            renderer.clearHighlightedSegment();
        }

        bool manualMergeHistoryChanged = false;
        if (uiActions.requestCancelMergeSelection)
        {
            session.selectedMergeTriangle.reset();
            renderer.clearHighlightedSegment();
        }
        if (uiActions.requestUndoManualMerge && !session.manualSegmentMerges.empty())
        {
            session.manualSegmentMerges.pop_back();
            session.selectedMergeTriangle.reset();
            renderer.clearHighlightedSegment();
            manualMergeHistoryChanged = true;
        }
        if (uiActions.requestResetManualMerges && !session.manualSegmentMerges.empty())
        {
            session.manualSegmentMerges.clear();
            session.selectedMergeTriangle.reset();
            renderer.clearHighlightedSegment();
            manualMergeHistoryChanged = true;
        }

        if (orbitDragging)
        {
            const double mouseDeltaX = mouseX - previousMouseX;
            const double mouseDeltaY = mouseY - previousMouseY;
            uiState.camera.yaw = wrapAngle(uiState.camera.yaw + static_cast<float>(mouseDeltaX) * kOrbitSensitivity);
            uiState.camera.pitch =
                clampOrbitPitch(uiState.camera.pitch + static_cast<float>(mouseDeltaY) * kOrbitSensitivity);
        }

        if (!mouseOverUi)
        {

            if (scroll != 0)
            {
                uiState.camera.distance = std::clamp(
                    uiState.camera.distance - static_cast<float>(scroll) * kZoomSensitivity,
                    kMinCameraDistance,
                    kMaxCameraDistance
                );
            }

            if ((uiState.manualSegmentMerge || !uiState.segmentation.automatic)
                && rightPressed && !previousRightPressed)
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

                const bgfx::Caps* caps = bgfx::getCaps();
                const auto pickedTriangle = caps == nullptr
                    ? std::nullopt
                    : segmesh::pickTriangleUnderCursor(
                        mesh,
                        mouseFramebufferX,
                        mouseFramebufferY,
                        static_cast<uint32_t>(width),
                        static_cast<uint32_t>(height),
                        modelRotation,
                        uiState.camera,
                        caps->homogeneousDepth
                    );
                if (!pickedTriangle.has_value())
                {
                    uiState.modelLoadError = "No triangle found under the cursor.";
                }
                else
                {
                    if (uiState.manualSegmentMerge)
                    {
                        const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
                        if (session.displayedTriangleLabels.size() != faceCount)
                        {
                            uiState.modelLoadError = "No segmentation preview is available to merge.";
                        }
                        else if (!session.selectedMergeTriangle.has_value())
                        {
                            const uint32_t selectedLabel =
                                session.displayedTriangleLabels[static_cast<std::size_t>(*pickedTriangle)];
                            std::string highlightError;
                            if (renderer.setHighlightedSegment(
                                    mesh,
                                    session.displayedTriangleLabels,
                                    selectedLabel,
                                    highlightError
                                ))
                            {
                                session.selectedMergeTriangle = *pickedTriangle;
                                uiState.modelLoadError.clear();
                            }
                            else
                            {
                                uiState.modelLoadError = highlightError;
                            }
                        }
                        else
                        {
                            const uint32_t sourceLabel = session.displayedTriangleLabels[
                                static_cast<std::size_t>(*session.selectedMergeTriangle)
                            ];
                            const uint32_t targetLabel = session.displayedTriangleLabels[
                                static_cast<std::size_t>(*pickedTriangle)
                            ];
                            if (sourceLabel == targetLabel)
                            {
                                session.selectedMergeTriangle.reset();
                                renderer.clearHighlightedSegment();
                                uiState.modelLoadError.clear();
                            }
                            else
                            {
                                std::vector<uint32_t> validationLabels = session.displayedTriangleLabels;
                                std::string mergeError;
                                if (segmesh::mergeAdjacentSegmentsByTriangles(
                                        mesh,
                                        *session.selectedMergeTriangle,
                                        *pickedTriangle,
                                        validationLabels,
                                        mergeError
                                    ))
                                {
                                    session.manualSegmentMerges.push_back({
                                        *session.selectedMergeTriangle,
                                        *pickedTriangle,
                                    });
                                    session.selectedMergeTriangle.reset();
                                    renderer.clearHighlightedSegment();
                                    manualMergeHistoryChanged = true;
                                    uiState.modelLoadError.clear();
                                }
                                else
                                {
                                    uiState.modelLoadError = mergeError;
                                }
                            }
                        }
                    }
                    else
                    {
                        if (!triangleAlreadySeeded(session.manualSeedTriangles, *pickedTriangle))
                        {
                            session.manualSegmentMerges.clear();
                            session.selectedMergeTriangle.reset();
                            renderer.clearHighlightedSegment();
                        }

                        std::string seedError;
                        if (addSeedTriangle(
                                renderer,
                                mesh,
                                *pickedTriangle,
                                session.manualSeedTriangles,
                                uiState,
                                session,
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
        }

        if (previewSettingsChanged || meshReloaded || activeSeedsChanged || mergeSettingsChanged
            || segmentationModelChanged || borderCurveSettingsChanged
            || manualMergeHistoryChanged)
        {
            std::string previewError;
            const std::vector<uint32_t>& currentSeeds =
                session.activeSeeds(uiState.segmentation.automatic);
            if (!refreshSegmentationPreview(
                    renderer,
                    mesh,
                    currentSeeds,
                    uiState,
                    session,
                    previewError
                ))
            {
                uiState.modelLoadError = previewError;
            }
            else if (manualMergeHistoryChanged)
            {
                uiState.modelLoadError.clear();
            }
        }

        if (uiActions.requestClearSeed)
        {
            renderer.clearSeedTriangle();
            session.manualSeedTriangles.clear();
            session.manualSegmentMerges.clear();
            session.selectedMergeTriangle.reset();
            renderer.clearHighlightedSegment();
            uiState.manualSegmentMerge = false;
            uiState.modelLoadError.clear();

            std::string previewError;
            if (!refreshSegmentationPreview(
                    renderer,
                    mesh,
                    session.manualSeedTriangles,
                    uiState,
                    session,
                    previewError
                ))
            {
                uiState.modelLoadError = previewError;
            }
        }

        if (uiActions.requestRandomSeed)
        {
            const auto randomSeed = pickRandomSeedTriangle(mesh, session.manualSeedTriangles, randomEngine);
            if (!randomSeed.has_value())
            {
                uiState.modelLoadError = "No unseeded triangles left. Use Clear seeds to reset.";
            }
            else
            {
                session.manualSegmentMerges.clear();
                session.selectedMergeTriangle.reset();
                renderer.clearHighlightedSegment();

                std::string seedError;
                if (addSeedTriangle(
                        renderer,
                        mesh,
                        *randomSeed,
                        session.manualSeedTriangles,
                        uiState,
                        session,
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

        if (session.displayedTriangleLabels.empty())
        {
            uiState.manualSegmentMerge = false;
            session.selectedMergeTriangle.reset();
            renderer.clearHighlightedSegment();
        }

        previousMouseX = mouseX;
        previousMouseY = mouseY;
        previousRightPressed = rightPressed;

        renderer.renderScene(
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            modelRotation,
            uiState.camera,
            uiState.rendering,
            uiState.borders
        );
        imguiEndFrame();
        renderer.frame();

        if (uiState.rendering.autoRotate)
        {
            modelRotation += uiState.rendering.rotateSpeed * dt;
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
