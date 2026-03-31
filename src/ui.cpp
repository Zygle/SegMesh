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
    ImGui::Separator();
    ImGui::Checkbox("Preview segmentation", &uiState.showSegmentation);
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

        if (uiState.automaticSegmentationMode == AutomaticSegmentationMode::Coarse)
        {
            ImGui::TextUnformatted("Coarse mode uses farthest-point geodesic seeding.");
        }
        else
        {
            ImGui::TextUnformatted("Fine mode uses feature-biased dense auto seeding.");
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
    if (!uiState.modelLoadError.empty())
    {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Model load error: %s", uiState.modelLoadError.c_str());
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

    return actions;
}
}
