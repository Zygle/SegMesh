#include "ui.h"
#include "imgui.h"

#include <algorithm>

namespace segmesh
{
RendererUiActions drawRendererPanel(
    const std::vector<std::filesystem::path>& modelPaths,
    int selectedModelIndex,
    const std::filesystem::path& objPath,
    uint32_t vertexCount,
    uint32_t triangleCount,
    uint32_t seedTriangleCount,
    const char* rendererName,
    RendererUiState& uiState
)
{
    RendererUiActions actions{};
    actions.pendingModelIndex = selectedModelIndex;

    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const float panelWidth = 315.0f;
    const float topMargin = 26.0f;
    const float sideMargin = 10.0f;
    const float rendererHeight = std::max(420.0f, displaySize.y - 74.0f);
    const float segmentationHeight = std::max(360.0f, displaySize.y - 118.0f);
    const float segmentationX = std::max(sideMargin, displaySize.x - panelWidth - 18.0f);

    ImGui::SetNextWindowPos(ImVec2(sideMargin, topMargin), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, rendererHeight), ImGuiCond_Once);
    ImGui::Begin("Renderer");

    const std::string activeModelLabel = objPath.filename().string();
    if (ImGui::BeginCombo("Model", activeModelLabel.c_str()))
    {
        for (int i = 0; i < static_cast<int>(modelPaths.size()); ++i)
        {
            const bool selected = i == selectedModelIndex;
            const std::string itemLabel = modelPaths[static_cast<std::size_t>(i)].filename().string();
            if (ImGui::Selectable(itemLabel.c_str(), selected))
            {
                actions.pendingModelIndex = i;
            }

            if (selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    ImGui::Text("Renderer: %s", rendererName);
    ImGui::Text("OBJ: %s", objPath.string().c_str());
    ImGui::Text("Available models: %u", static_cast<uint32_t>(modelPaths.size()));
    ImGui::Text("Vertices: %u", vertexCount);
    ImGui::Text("Triangles: %u", triangleCount);
    if (!uiState.modelLoadError.empty())
    {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Status: %s", uiState.modelLoadError.c_str());
    }
    ImGui::Separator();
    ImGui::Checkbox("Auto rotate", &uiState.autoRotate);
    ImGui::SliderFloat("Rotate speed", &uiState.rotateSpeed, 0.0f, 3.0f);
    ImGui::SliderFloat("Camera distance", &uiState.cameraDistance, 1.0f, 10.0f);
    ImGui::SliderFloat("Camera yaw", &uiState.cameraYaw, -3.14f, 3.14f);
    ImGui::SliderFloat("Camera pitch", &uiState.cameraPitch, -1.2f, 1.2f);
    ImGui::Separator();
    ImGui::ColorEdit3("Base color", uiState.baseColor);
    ImGui::SliderFloat3("Light direction", uiState.lightDirection, -1.0f, 1.0f);
    ImGui::ColorEdit3("Light color", uiState.lightColor);
    ImGui::SliderFloat("Light intensity", &uiState.lightIntensity, 0.0f, 4.0f);
    ImGui::SliderFloat("Ambient", &uiState.ambientStrength, 0.0f, 1.0f);
    ImGui::SliderFloat("Specular", &uiState.specularStrength, 0.0f, 1.0f);
    ImGui::SliderFloat("Shininess", &uiState.shininess, 8.0f, 256.0f);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(segmentationX, topMargin), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, segmentationHeight), ImGuiCond_Once);
    ImGui::Begin("Segmentation");

    ImGui::Checkbox("Preview segmentation", &uiState.showSegmentation);
    ImGui::Checkbox("Black segment borders", &uiState.showSegmentBorders);
    int segmentationModel = static_cast<int>(uiState.segmentationModelType);
    const char* segmentationModelLabels[] = {"Graphical", "Engineering"};
    ImGui::TextUnformatted("Segmentation model");
    ImGui::Combo("##SegmentationModel", &segmentationModel, segmentationModelLabels, IM_ARRAYSIZE(segmentationModelLabels));
    uiState.segmentationModelType = static_cast<SegmentationModelType>(segmentationModel);
    ImGui::Separator();
    ImGui::Checkbox("Automatic segmentation", &uiState.automaticSegmentation);
    if (uiState.automaticSegmentation)
    {
        int autoMode = static_cast<int>(uiState.automaticSegmentationMode);
        const char* autoModeLabels[] = {"Coarse", "Fine"};
        ImGui::Combo("Auto mode", &autoMode, autoModeLabels, IM_ARRAYSIZE(autoModeLabels));
        uiState.automaticSegmentationMode = static_cast<AutomaticSegmentationMode>(autoMode);

        const int minSeedCount =
            uiState.automaticSegmentationMode == AutomaticSegmentationMode::Fine ? 20 : 1;
        const int maxSeedCount =
            uiState.automaticSegmentationMode == AutomaticSegmentationMode::Fine ? 200 : 64;
        uiState.automaticSeedCount = std::clamp(uiState.automaticSeedCount, minSeedCount, maxSeedCount);
        ImGui::SliderInt("Auto seed target", &uiState.automaticSeedCount, minSeedCount, maxSeedCount);
        ImGui::Checkbox("Step-by-step", &uiState.stepAutomaticSegmentation);
        if (uiState.stepAutomaticSegmentation)
        {
            uiState.automaticStepSeedCount = std::max(uiState.automaticStepSeedCount, 1);
            if (ImGui::Button("Next step"))
            {
                actions.requestAutomaticStep = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset steps"))
            {
                actions.requestAutomaticStepReset = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Finish"))
            {
                actions.requestAutomaticStepFinish = true;
            }
            ImGui::Text("Active auto seeds: %d", uiState.automaticStepSeedCount);
        }

        if (uiState.automaticSegmentationMode == AutomaticSegmentationMode::Coarse)
        {
            ImGui::TextUnformatted("Coarse mode uses farthest-point geodesic seeding.");
        }
        else
        {
            ImGui::TextUnformatted("Fine mode uses feature-biased dense auto seeding.");
            ImGui::Checkbox("Merge fine segments", &uiState.mergeFineSegments);
            if (uiState.mergeFineSegments)
            {
                const int maxMergedSegmentCount = std::max(uiState.automaticSeedCount, 1);
                uiState.mergeTargetSegmentCount =
                    std::clamp(uiState.mergeTargetSegmentCount, 1, maxMergedSegmentCount);
                ImGui::SliderInt("Target regions", &uiState.mergeTargetSegmentCount, 1, maxMergedSegmentCount);
                ImGui::SliderFloat("Merge cost limit", &uiState.mergeCostThreshold, 0.05f, 2.0f);
            }
            ImGui::Checkbox("Cleanup small segments", &uiState.cleanupSmallFineSegments);
            if (uiState.cleanupSmallFineSegments)
            {
                uiState.minFineSegmentTriangles = std::clamp(uiState.minFineSegmentTriangles, 2, 128);
                ImGui::SliderInt("Min triangles", &uiState.minFineSegmentTriangles, 2, 128);
            }
            ImGui::TextUnformatted("Lower merge cost means a weaker boundary, so it merges first.");
        }
    }
    if (uiState.showSegmentation && !uiState.automaticSegmentation && seedTriangleCount == 0)
    {
        ImGui::TextUnformatted("Add at least one seed triangle to solve the segmentation.");
    }
    ImGui::Separator();
    ImGui::BeginDisabled(uiState.automaticSegmentation);
    if (ImGui::Button("Place random seed"))
    {
        actions.requestRandomSeed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear seeds"))
    {
        actions.requestClearSeed = true;
    }
    ImGui::EndDisabled();
    ImGui::Text("Seed count: %u", seedTriangleCount);
    if (uiState.automaticSegmentation)
    {
        ImGui::TextUnformatted("Manual seed placement is disabled while automatic mode is enabled.");
    }
    ImGui::End();

    return actions;
}
}
