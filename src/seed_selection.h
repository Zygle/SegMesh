#pragma once

#include "mesh_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace segmesh
{
// Both generators start with one seed per connected face component. When supplied,
// outMinimumSeedCount receives that prefix length on success, or zero on failure.
bool selectAutomaticSeedsCoarse(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t approximateSeedCount,
    std::vector<uint32_t>& outSeedTriangles,
    std::string& error,
    uint32_t* outMinimumSeedCount = nullptr
);

bool selectAutomaticSeedsFine(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t approximateSeedCount,
    std::vector<uint32_t>& outSeedTriangles,
    std::string& error,
    uint32_t* outMinimumSeedCount = nullptr
);
}
