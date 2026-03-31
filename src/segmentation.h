#pragma once

#include "mesh_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace segmesh
{
bool selectAutomaticSeedsCoarse(
    const CpuMesh& mesh,
    uint32_t approximateSeedCount,
    std::vector<uint32_t>& outSeedTriangles,
    std::string& error
);

bool selectAutomaticSeedsFine(
    const CpuMesh& mesh,
    uint32_t approximateSeedCount,
    std::vector<uint32_t>& outSeedTriangles,
    std::string& error
);

bool segmentMeshRandomWalk(
    const CpuMesh& mesh,
    const std::vector<uint32_t>& seedTriangles,
    std::vector<uint32_t>& outTriangleLabels,
    std::string& error
);
}
