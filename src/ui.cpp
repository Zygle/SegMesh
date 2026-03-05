#include "ui.h"
#include "imgui.h"

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
    if (ImGui::Button("Place random seed"))
    {
        actions.requestRandomSeed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear seeds"))
    {
        actions.requestClearSeed = true;
    }
    ImGui::Text("Seed count: %u", seedTriangleCount);
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
