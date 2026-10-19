#pragma once

#include "camera.h"
#include "mesh_types.h"

#include <cstdint>
#include <optional>

namespace segmesh
{
std::optional<uint32_t> pickTriangleUnderCursor(
    const CpuMesh& mesh,
    double mouseFramebufferX,
    double mouseFramebufferY,
    uint32_t width,
    uint32_t height,
    float modelRotation,
    const CameraSettings& cameraSettings,
    bool homogeneousDepth
);
}
