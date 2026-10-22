#pragma once

#include "render_settings.h"
#include "segmentation_settings.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace segmesh
{
struct RendererUiState
{
    CameraSettings camera;
    RenderSettings rendering;
    BorderSettings borders;
    SegmentationSettings segmentation;

    bool manualSegmentMerge = false;
    std::string modelLoadError;
};

struct RendererUiActions
{
    int pendingModelIndex = 0;
    bool requestRandomSeed = false;
    bool requestClearSeed = false;
    bool requestAutomaticStep = false;
    bool requestAutomaticStepReset = false;
    bool requestAutomaticStepFinish = false;
    bool requestCancelMergeSelection = false;
    bool requestUndoManualMerge = false;
    bool requestResetManualMerges = false;
};

RendererUiActions drawRendererPanel(
    const std::vector<std::filesystem::path>& modelPaths,
    int selectedModelIndex,
    const std::filesystem::path& objPath,
    uint32_t vertexCount,
    uint32_t triangleCount,
    uint32_t seedTriangleCount,
    int automaticStepSeedCount,
    bool hasSegmentationPreview,
    int32_t selectedMergeSegment,
    uint32_t manualMergeCount,
    const char* rendererName,
    RendererUiState& uiState
);
}
