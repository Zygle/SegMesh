#include "select.h"

#include <bx/math.h>

#include <cmath>
#include <limits>

namespace segmesh
{
namespace
{
constexpr float kPickEpsilon = 1.0e-8f;

Float3 subtract(const Float3& a, const Float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Float3 scale(const Float3& v, float scalar)
{
    return {v.x * scalar, v.y * scalar, v.z * scalar};
}

float dot(const Float3& a, const Float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Float3 cross(const Float3& a, const Float3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float length(const Float3& v)
{
    return std::sqrt(dot(v, v));
}

Float3 normalize(const Float3& v, const Float3& fallback)
{
    const float len = length(v);
    if (len < kPickEpsilon)
    {
        return fallback;
    }

    return scale(v, 1.0f / len);
}

Float3 toFloat3(const bx::Vec3& v)
{
    return {v.x, v.y, v.z};
}

bool rayTriangleIntersection(
    const Float3& rayOrigin,
    const Float3& rayDirection,
    const Float3& p0,
    const Float3& p1,
    const Float3& p2,
    float& outDistance
)
{
    const Float3 edge01 = subtract(p1, p0);
    const Float3 edge02 = subtract(p2, p0);
    const Float3 p = cross(rayDirection, edge02);
    const float determinant = dot(edge01, p);
    if (std::abs(determinant) < kPickEpsilon)
    {
        return false;
    }

    const float inverseDeterminant = 1.0f / determinant;
    const Float3 t = subtract(rayOrigin, p0);
    const float u = dot(t, p) * inverseDeterminant;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }

    const Float3 q = cross(t, edge01);
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
}

std::optional<uint32_t> pickTriangleUnderCursor(
    const CpuMesh& mesh,
    double mouseFramebufferX,
    double mouseFramebufferY,
    uint32_t width,
    uint32_t height,
    float modelRotation,
    const CameraSettings& cameraSettings,
    bool homogeneousDepth
)
{
    if (width == 0 || height == 0)
    {
        return std::nullopt;
    }

    const CameraMatrices cameraMatrices = buildCameraMatrices(
        cameraSettings,
        width,
        height,
        homogeneousDepth
    );

    float invView[16];
    float invProj[16];
    float model[16];
    float invModel[16];
    bx::mtxInverse(invView, cameraMatrices.view.data());
    bx::mtxInverse(invProj, cameraMatrices.projection.data());
    bx::mtxRotateY(model, modelRotation);
    bx::mtxInverse(invModel, model);

    const float ndcX = static_cast<float>(((mouseFramebufferX + 0.5) / static_cast<double>(width)) * 2.0 - 1.0);
    const float ndcY = static_cast<float>(1.0 - ((mouseFramebufferY + 0.5) / static_cast<double>(height)) * 2.0);
    const float nearClipZ = homogeneousDepth ? -1.0f : 0.0f;

    const bx::Vec3 nearClip{ndcX, ndcY, nearClipZ};
    const bx::Vec3 farClip{ndcX, ndcY, 1.0f};
    const bx::Vec3 nearWorld = bx::mulH(bx::mulH(nearClip, invProj), invView);
    const bx::Vec3 farWorld = bx::mulH(bx::mulH(farClip, invProj), invView);

    const Float3 rayOrigin = toFloat3(bx::mulH(nearWorld, invModel));
    const Float3 rayEnd = toFloat3(bx::mulH(farWorld, invModel));
    const Float3 rayDirection = normalize(subtract(rayEnd, rayOrigin), {0.0f, 0.0f, -1.0f});

    float closestDistance = std::numeric_limits<float>::max();
    std::optional<uint32_t> closestTriangle;
    const uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    for (uint32_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex)
    {
        const uint32_t indexOffset = triangleIndex * 3;
        const MeshVertex& v0 = mesh.vertices[mesh.indices[indexOffset + 0]];
        const MeshVertex& v1 = mesh.vertices[mesh.indices[indexOffset + 1]];
        const MeshVertex& v2 = mesh.vertices[mesh.indices[indexOffset + 2]];

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
}
