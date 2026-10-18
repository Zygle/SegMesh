#pragma once

#include "mesh_types.h"
#include "seed_selection.h"
#include "segment_merging.h"

#include <cstdint>
#include <string>
#include <vector>

namespace segmesh
{
bool segmentMeshRandomWalk(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    const std::vector<uint32_t>& seedTriangles,
    std::vector<uint32_t>& outTriangleLabels,
    std::vector<float>& outTriangleConfidence,
    std::string& error
);
}
