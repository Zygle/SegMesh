#pragma once

#include "render_settings.h"

#include <array>
#include <cstdint>

namespace segmesh
{
struct CameraMatrices
{
    std::array<float, 3> eyePosition{};
    std::array<float, 16> view{};
    std::array<float, 16> projection{};
};

CameraMatrices buildCameraMatrices(
    const CameraSettings& settings,
    uint32_t width,
    uint32_t height,
    bool homogeneousDepth
);
}
