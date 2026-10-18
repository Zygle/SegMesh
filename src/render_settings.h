#pragma once

namespace segmesh
{
enum class SegmentBorderDisplayMode : int
{
    Original = 0,
    Smoothed = 1,
    Both = 2,
};

struct CameraSettings
{
    float distance = 3.0f;
    float yaw = 0.0f;
    float pitch = 0.35f;
};

struct RenderSettings
{
    float rotateSpeed = 0.8f;
    bool autoRotate = false;

    float baseColor[3] = {0.78f, 0.78f, 0.78f};
    float lightDirection[3] = {-0.45f, -0.9f, -0.25f};
    float lightColor[3] = {1.0f, 0.98f, 0.95f};
    float lightIntensity = 1.5f;
    float ambientStrength = 0.22f;
    float specularStrength = 0.32f;
    float shininess = 64.0f;

    bool showSegmentation = false;
    bool shadeByConfidence = false;
};

struct BorderSettings
{
    bool visible = true;
    SegmentBorderDisplayMode displayMode = SegmentBorderDisplayMode::Both;
    float fitTolerance = 0.01f;
    float drawingTolerance = 0.1f;
    bool showControlPolygon = false;
};
}
