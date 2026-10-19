#include "camera.h"

#include <bx/math.h>

#include <cmath>

namespace segmesh
{
CameraMatrices buildCameraMatrices(
    const CameraSettings& settings,
    uint32_t width,
    uint32_t height,
    bool homogeneousDepth
)
{
    CameraMatrices matrices{};
    const float cosPitch = std::cos(settings.pitch);
    matrices.eyePosition = {
        std::sin(settings.yaw) * cosPitch * settings.distance,
        std::sin(settings.pitch) * settings.distance,
        std::cos(settings.yaw) * cosPitch * settings.distance,
    };

    const bx::Vec3 eye{matrices.eyePosition[0], matrices.eyePosition[1], matrices.eyePosition[2]};
    const bx::Vec3 at{0.0f, 0.0f, 0.0f};
    bx::mtxLookAt(matrices.view.data(), eye, at);

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    bx::mtxProj(matrices.projection.data(), 60.0f, aspect, 0.01f, 100.0f, homogeneousDepth);
    return matrices;
}
}
