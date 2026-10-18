#pragma once

#include "mesh_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Shared implementation details for the segmentation stages.
namespace segmesh::detail
{
struct FaceGraphEdgeInfo
{
    uint32_t face0 = 0;
    uint32_t face1 = 0;
    double edgeLength = 0.0;
    double difference = 0.0;
};

double faceNormalDot(const CpuMesh& mesh, uint32_t faceIndex, uint32_t neighborIndex);

double edgeDifferenceForMode(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t faceIndex,
    std::size_t edgeSlot,
    uint32_t neighborIndex
);

bool buildFaceGraphEdges(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    std::vector<FaceGraphEdgeInfo>& outEdges,
    double& outAverageDifference,
    std::string& error
);
}
