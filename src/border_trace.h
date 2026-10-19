#pragma once

#include "mesh_types.h"

#include <Eigen/Core>

#include <compare>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace segmesh::detail
{
struct LabelPair
{
    uint32_t first = 0;
    uint32_t second = 0;

    auto operator<=>(const LabelPair&) const = default;
};

struct TracedChain
{
    LabelPair labels;
    std::vector<uint32_t> vertexIndices;
    bool closedLoopArc = false;
    bool hasEndpointTangents = false;
    Eigen::Vector3d startTangent{};
    Eigen::Vector3d endTangent{};
};

// The caller must validate the label count and mesh topology array sizes first.
// Outputs are reset on entry and populated only on success.
bool traceSegmentBorders(
    const CpuMesh& mesh,
    std::span<const uint32_t> triangleLabels,
    std::vector<TracedChain>& outChains,
    uint32_t& outBoundaryEdgeCount,
    std::string& error
);
}
