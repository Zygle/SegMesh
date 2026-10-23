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
    const int automaticStepSeedCount,
    bool hasSegmentationPreview,
    int32_t selectedMergeSegment,
    uint32_t manualMergeCount,
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
    ImGui::Checkbox("Auto rotate", &uiState.rendering.autoRotate);
    ImGui::SliderFloat("Rotate speed", &uiState.rendering.rotateSpeed, 0.0f, 3.0f);
    ImGui::SliderFloat("Camera distance", &uiState.camera.distance, 1.0f, 10.0f);
    ImGui::SliderFloat("Camera yaw", &uiState.camera.yaw, -3.14f, 3.14f);
    ImGui::SliderFloat("Camera pitch", &uiState.camera.pitch, -1.2f, 1.2f);
    ImGui::Separator();
    ImGui::ColorEdit3("Base color", uiState.rendering.baseColor);
    ImGui::SliderFloat3("Light direction", uiState.rendering.lightDirection, -1.0f, 1.0f);
    ImGui::ColorEdit3("Light color", uiState.rendering.lightColor);
    ImGui::SliderFloat("Light intensity", &uiState.rendering.lightIntensity, 0.0f, 4.0f);
    ImGui::SliderFloat("Ambient", &uiState.rendering.ambientStrength, 0.0f, 1.0f);
    ImGui::SliderFloat("Specular", &uiState.rendering.specularStrength, 0.0f, 1.0f);
    ImGui::SliderFloat("Shininess", &uiState.rendering.shininess, 8.0f, 256.0f);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(segmentationX, topMargin), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, segmentationHeight), ImGuiCond_Once);
    ImGui::Begin("Segmentation");

    ImGui::Checkbox("Preview segmentation", &uiState.rendering.showSegmentation);
    ImGui::BeginDisabled(!uiState.rendering.showSegmentation);
    ImGui::Checkbox("Shade by confidence", &uiState.rendering.shadeByConfidence);
    ImGui::EndDisabled();
    ImGui::Checkbox("Segment borders", &uiState.borders.visible);
    if (uiState.borders.visible)
    {
        int borderDisplayMode = static_cast<int>(uiState.borders.displayMode);
        const char* borderDisplayModeLabels[] = {"Original", "Smoothed", "Both"};
        ImGui::Combo(
            "Border style",
            &borderDisplayMode,
            borderDisplayModeLabels,
            IM_ARRAYSIZE(borderDisplayModeLabels)
        );
        uiState.borders.displayMode = static_cast<SegmentBorderDisplayMode>(borderDisplayMode);

        if (uiState.borders.displayMode != SegmentBorderDisplayMode::Original)
        {
            ImGui::SliderFloat(
                "Fit tolerance",
                &uiState.borders.fitTolerance,
                0.001f,
                0.05f,
                "%.4f",
                ImGuiSliderFlags_Logarithmic
            );
            ImGui::SliderFloat(
                "Drawing tolerance",
                &uiState.borders.drawingTolerance,
                0.0001f,
                0.01f,
                "%.4f",
                ImGuiSliderFlags_Logarithmic
            );
            ImGui::Checkbox("Show B\xC3\xA9zier control polygons", &uiState.borders.showControlPolygon);
            ImGui::TextUnformatted("Cyan: piecewise cubic B\xC3\xA9zier border.");
            ImGui::TextUnformatted("Orange: cubic control polygons.");
        }
    }
    int segmentationModel = static_cast<int>(uiState.segmentation.model);
    const char* segmentationModelLabels[] = {"Graphical", "Engineering"};
    ImGui::TextUnformatted("Segmentation model");
    ImGui::Combo("##SegmentationModel", &segmentationModel, segmentationModelLabels, IM_ARRAYSIZE(segmentationModelLabels));
    uiState.segmentation.model = static_cast<SegmentationModelType>(segmentationModel);
    if (uiState.segmentation.model == SegmentationModelType::Engineering)
    {
        ImGui::TextUnformatted("Engineering mode uses approximate curvature heuristics.");
    }
    ImGui::Separator();
    ImGui::Checkbox("Automatic segmentation", &uiState.segmentation.automatic);
    if (uiState.segmentation.automatic)
    {
        int autoMode = static_cast<int>(uiState.segmentation.automaticMode);
        const char* autoModeLabels[] = {"Coarse", "Fine"};
        ImGui::Combo("Auto mode", &autoMode, autoModeLabels, IM_ARRAYSIZE(autoModeLabels));
        uiState.segmentation.automaticMode = static_cast<AutomaticSegmentationMode>(autoMode);

        const int minSeedCount =
            uiState.segmentation.automaticMode == AutomaticSegmentationMode::Fine ? 20 : 1;
        const int maxSeedCount =
            uiState.segmentation.automaticMode == AutomaticSegmentationMode::Fine ? 200 : 64;
        uiState.segmentation.automaticSeedCount = std::clamp(uiState.segmentation.automaticSeedCount, minSeedCount, maxSeedCount);
        ImGui::SliderInt("Auto seed target", &uiState.segmentation.automaticSeedCount, minSeedCount, maxSeedCount);
        ImGui::Checkbox("Step-by-step", &uiState.segmentation.stepByStep);
        if (uiState.segmentation.stepByStep)
        {
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
            ImGui::Text("Active auto seeds: %d", std::max(automaticStepSeedCount, 1));
            ImGui::TextWrapped("Each disconnected part starts with a seed.");
        }

        if (uiState.segmentation.automaticMode == AutomaticSegmentationMode::Coarse)
        {
            ImGui::TextUnformatted("Coarse mode uses farthest-point geodesic seeding.");
        }
        else
        {
            ImGui::TextUnformatted("Fine mode uses feature-biased dense auto seeding.");
            ImGui::Checkbox("Merge fine segments", &uiState.segmentation.mergeFineSegments);
            if (uiState.segmentation.mergeFineSegments)
            {
                const int maxMergedSegmentCount = std::max(uiState.segmentation.automaticSeedCount, 1);
                uiState.segmentation.mergeTargetSegmentCount =
                    std::clamp(uiState.segmentation.mergeTargetSegmentCount, 1, maxMergedSegmentCount);
                ImGui::SliderInt("Target regions", &uiState.segmentation.mergeTargetSegmentCount, 1, maxMergedSegmentCount);
                ImGui::Checkbox("Force target regions", &uiState.segmentation.forceMergeTargetSegmentCount);
                if (!uiState.segmentation.forceMergeTargetSegmentCount)
                {
                    ImGui::SliderFloat("Merge cost limit", &uiState.segmentation.mergeCostThreshold, 0.05f, 2.0f);
                }
            }
            ImGui::Checkbox("Cleanup small segments", &uiState.segmentation.cleanupSmallFineSegments);
            if (uiState.segmentation.cleanupSmallFineSegments)
            {
                uiState.segmentation.minFineSegmentTriangles = std::clamp(uiState.segmentation.minFineSegmentTriangles, 2, 128);
                ImGui::SliderInt("Min triangles", &uiState.segmentation.minFineSegmentTriangles, 2, 128);
            }
            if (uiState.segmentation.forceMergeTargetSegmentCount)
            {
                ImGui::TextUnformatted("Force target can merge across strong boundaries to reach the requested count.");
            }
            else
            {
                ImGui::TextUnformatted("Lower merge cost means a weaker boundary, so it merges first.");
            }
        }
    }
    if (uiState.rendering.showSegmentation && !uiState.segmentation.automatic && seedTriangleCount == 0)
    {
        ImGui::TextUnformatted("Add at least one seed triangle to solve the segmentation.");
    }
    ImGui::Separator();
    if (!hasSegmentationPreview)
    {
        uiState.manualSegmentMerge = false;
    }
    ImGui::BeginDisabled(!hasSegmentationPreview);
    ImGui::Checkbox("Manual segment merge", &uiState.manualSegmentMerge);
    ImGui::EndDisabled();
    if (uiState.manualSegmentMerge)
    {
        ImGui::TextWrapped("Right-click a source segment, then right-click a neighboring target segment.");
        if (selectedMergeSegment >= 0)
        {
            ImGui::Text("Selected source: %d", selectedMergeSegment);
        }
        else
        {
            ImGui::TextUnformatted("Selected source: none");
        }

        ImGui::BeginDisabled(selectedMergeSegment < 0);
        if (ImGui::Button("Cancel selection"))
        {
            actions.requestCancelMergeSelection = true;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(manualMergeCount == 0);
        if (ImGui::Button("Undo merge"))
        {
            actions.requestUndoManualMerge = true;
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(manualMergeCount == 0);
        if (ImGui::Button("Reset manual merges"))
        {
            actions.requestResetManualMerges = true;
        }
        ImGui::EndDisabled();
        ImGui::Text("Manual merge count: %u", manualMergeCount);
    }

    ImGui::Separator();
    ImGui::BeginDisabled(uiState.segmentation.automatic || uiState.manualSegmentMerge);
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
    if (uiState.segmentation.automatic)
    {
        ImGui::TextUnformatted("Manual seed placement is disabled while automatic mode is enabled.");
    }
    else if (uiState.manualSegmentMerge)
    {
        ImGui::TextUnformatted("Seed placement is disabled while manual merge mode is enabled.");
    }
    ImGui::End();

    return actions;
}
}
