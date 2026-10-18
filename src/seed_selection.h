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
}
