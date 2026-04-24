#pragma once

#include "mesh_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace segmesh
{
bool selectAutomaticSeedsCoarse(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t approximateSeedCount,
    std::vector<uint32_t>& outSeedTriangles,
    std::string& error
);

bool selectAutomaticSeedsFine(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t approximateSeedCount,
    std::vector<uint32_t>& outSeedTriangles,
    std::string& error
);

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

bool segmentMeshRandomWalk(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    const std::vector<uint32_t>& seedTriangles,
    std::vector<uint32_t>& outTriangleLabels,
    std::string& error
);
}
