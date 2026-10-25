#pragma once

#include "mesh_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace segmesh
{
bool mergeSegmentsByBoundaryCost(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t targetSegmentCount,
    double maxRelativeMergeCost,
    std::vector<uint32_t>& inOutTriangleLabels,
    std::string& error
);

bool mergeSmallSegmentsByTriangleCount(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t minTriangleCount,
    std::vector<uint32_t>& inOutTriangleLabels,
    std::string& error
);

bool mergeAdjacentSegmentsByTriangles(
    const CpuMesh& mesh,
    uint32_t sourceTriangle,
    uint32_t targetTriangle,
    std::vector<uint32_t>& inOutTriangleLabels,
    std::string& error
);
}
