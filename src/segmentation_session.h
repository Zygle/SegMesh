#pragma once

#include "segmentation_settings.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace segmesh
{
struct CpuMesh;

struct ManualSegmentMerge
{
    uint32_t sourceTriangle = 0;
    uint32_t targetTriangle = 0;
};

class SegmentationSession
{
public:
    std::vector<uint32_t> manualSeedTriangles;
    std::vector<uint32_t> automaticSeedPlanTriangles;
    std::vector<uint32_t> automaticSeedTriangles;
    std::vector<ManualSegmentMerge> manualSegmentMerges;
    std::vector<uint32_t> displayedTriangleLabels;
    std::optional<uint32_t> selectedMergeTriangle;

    const std::vector<uint32_t>& activeSeeds(bool automatic) const;
    void resetForMesh();
    bool solveRawLabels(
        const CpuMesh& mesh,
        SegmentationModelType segmentationModelType,
        const std::vector<uint32_t>& seedTriangles,
        std::vector<uint32_t>& outTriangleLabels,
        std::vector<float>& outTriangleConfidence,
        std::string& error
    );

private:
    struct RandomWalkLabelCache
    {
        bool valid = false;
        uint64_t meshGeneration = 0;
        segmesh::SegmentationModelType segmentationModelType = segmesh::SegmentationModelType::Graphical;
        std::vector<uint32_t> seedTriangles;
        std::vector<uint32_t> unmergedTriangleLabels;
        std::vector<float> triangleConfidence;

        void invalidate()
        {
            valid = false;
            seedTriangles.clear();
            unmergedTriangleLabels.clear();
            triangleConfidence.clear();
        }

        bool matches(
            uint64_t requestedMeshGeneration,
            segmesh::SegmentationModelType requestedSegmentationModelType,
            const std::vector<uint32_t>& requestedSeedTriangles,
            uint32_t faceCount
        ) const
        {
            return valid
                && meshGeneration == requestedMeshGeneration
                && segmentationModelType == requestedSegmentationModelType
                && seedTriangles == requestedSeedTriangles
                && unmergedTriangleLabels.size() == faceCount
                && triangleConfidence.size() == faceCount;
        }
    };

    uint64_t meshGeneration_ = 1;
    RandomWalkLabelCache randomWalkLabelCache_;
};
}
