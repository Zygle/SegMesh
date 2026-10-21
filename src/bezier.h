#pragma once

#include "mesh_types.h"

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace segmesh
{
struct BorderCurveFitStats
{
    double maxError = 0.0;
    double rmsError = 0.0;
};

Float3 evaluateBezier(std::span<const Float3> controlPoints, double t);

bool fitEndpointConstrainedCubic(
    std::span<const Float3> points,
    std::array<Float3, 4>& outControlPoints,
    BorderCurveFitStats& outStats,
    std::string& error
);

namespace detail
{
struct CubicPiece
{
    std::array<Float3, 4> controlPoints{};
    std::size_t firstSample = 0;
    std::size_t lastSample = 0;
    BorderCurveFitStats fitStats{};
};

// Internal helpers shared by curve fitting and border drawing.
std::vector<Float3> removeConsecutiveCoincidentPoints(
    std::span<const Float3> points
);

Eigen::Vector3d forwardTangentAt(
    std::span<const Float3> points,
    std::size_t index
);

double cubicFlatness(const std::array<Float3, 4>& cubic);

void splitCubicAtHalf(
    const std::array<Float3, 4>& cubic,
    std::array<Float3, 4>& left,
    std::array<Float3, 4>& right
);

// Expects a finite non-negative tolerance and outSubdivisionCapped initialized to false.
// Preserves both endpoints; sets the flag if the depth limit prevents meeting tolerance.
std::vector<Float3> flattenCubic(
    const std::array<Float3, 4>& cubic,
    double tolerance,
    bool& outSubdivisionCapped
);

// Expects at least two cleaned samples and an empty output vector.
bool fitAdaptiveCubicPieces(
    std::span<const Float3> points,
    double fitTolerance,
    const Eigen::Vector3d& chainStartTangent,
    const Eigen::Vector3d& chainEndTangent,
    std::vector<CubicPiece>& outPieces,
    std::string& error
);
}
}
