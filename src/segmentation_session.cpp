#include "segmentation_session.h"
#include "segmentation.h"

namespace segmesh
{
const std::vector<uint32_t>& SegmentationSession::activeSeeds(bool automatic) const
{
    return automatic ? automaticSeedTriangles : manualSeedTriangles;
}

void SegmentationSession::resetForMesh()
{
    manualSeedTriangles.clear();
    automaticSeedPlanTriangles.clear();
    automaticSeedTriangles.clear();
    manualSegmentMerges.clear();
    displayedTriangleLabels.clear();
    selectedMergeTriangle.reset();
    ++meshGeneration_;
    randomWalkLabelCache_.invalidate();
}

bool SegmentationSession::solveRawLabels(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    const std::vector<uint32_t>& seedTriangles,
    std::vector<uint32_t>& outTriangleLabels,
    std::vector<float>& outTriangleConfidence,
    std::string& error
)
{
    const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (randomWalkLabelCache_.matches(meshGeneration_, segmentationModelType, seedTriangles, faceCount))
    {
        outTriangleLabels = randomWalkLabelCache_.unmergedTriangleLabels;
        outTriangleConfidence = randomWalkLabelCache_.triangleConfidence;
    }
    else
    {
        if (!segmesh::segmentMeshRandomWalk(
                mesh,
                segmentationModelType,
                seedTriangles,
                outTriangleLabels,
                outTriangleConfidence,
                error
            ))
        {
            randomWalkLabelCache_.invalidate();
            return false;
        }

        randomWalkLabelCache_.valid = true;
        randomWalkLabelCache_.meshGeneration = meshGeneration_;
        randomWalkLabelCache_.segmentationModelType = segmentationModelType;
        randomWalkLabelCache_.seedTriangles = seedTriangles;
        randomWalkLabelCache_.unmergedTriangleLabels = outTriangleLabels;
        randomWalkLabelCache_.triangleConfidence = outTriangleConfidence;
    }

    return true;
}
}
