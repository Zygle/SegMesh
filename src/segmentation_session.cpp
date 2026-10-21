#include "segmentation_session.h"
#include "segmentation.h"

#include <algorithm>
#include <cstddef>
#include <limits>

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
    borderCurveCache_.invalidate();
}

bool SegmentationSession::buildBorderOverlay(
    const CpuMesh& mesh,
    std::span<const uint32_t> triangleLabels,
    const BorderCurveSettings& settings,
    BorderCurveOverlay& outOverlay,
    std::string& error
)
{
    return borderCurveCache_.buildOverlay(
        mesh, meshGeneration_, triangleLabels, settings, outOverlay, error
    );
}

bool SegmentationSession::buildAutomaticSeedPlan(
    const CpuMesh& mesh,
    const SegmentationSettings& settings,
    std::string& error
)
{
    const int minSeedCount = settings.automaticMode == AutomaticSegmentationMode::Fine ? 20 : 1;
    const uint32_t targetSeedCount = static_cast<uint32_t>(std::max(settings.automaticSeedCount, minSeedCount));

    bool success = false;
    if (settings.automaticMode == AutomaticSegmentationMode::Coarse)
    {
        success = segmesh::selectAutomaticSeedsCoarse(
            mesh,
            settings.model,
            targetSeedCount,
            automaticSeedPlanTriangles,
            error
        );
    }
    else
    {
        success = segmesh::selectAutomaticSeedsFine(
            mesh,
            settings.model,
            targetSeedCount,
            automaticSeedPlanTriangles,
            error
        );
    }

    if (!success)
    {
        automaticSeedPlanTriangles.clear();
        return false;
    }

    return true;
}

bool SegmentationSession::selectAutomaticSeedPrefix(
    const SegmentationSettings& settings,
    int& automaticStepSeedCount,
    std::string& error
)
{
    const uint32_t planSeedCount = static_cast<uint32_t>(automaticSeedPlanTriangles.size());
    if (planSeedCount == 0)
    {
        automaticStepSeedCount = 0;
        automaticSeedTriangles.clear();
        error = "Automatic segmentation has no generated seed plan.";
        return false;
    }

    uint32_t activeSeedCount = planSeedCount;
    if (settings.stepByStep)
    {
        const uint32_t requestedSeedCount = static_cast<uint32_t>(std::max(automaticStepSeedCount, 1));
        activeSeedCount = std::min(requestedSeedCount, planSeedCount);
    }

    automaticStepSeedCount = static_cast<int>(activeSeedCount);
    automaticSeedTriangles.assign(
        automaticSeedPlanTriangles.begin(),
        automaticSeedPlanTriangles.begin() + static_cast<std::ptrdiff_t>(activeSeedCount)
    );
    return true;
}

bool SegmentationSession::buildPreviewLabels(
    const CpuMesh& mesh,
    const SegmentationSettings& settings,
    const std::vector<uint32_t>& seedTriangles,
    std::vector<uint32_t>& outTriangleLabels,
    std::vector<float>& outTriangleConfidence,
    uint32_t& outSegmentCount,
    std::string& error
)
{
    if (!solveRawLabels(
            mesh,
            settings.model,
            seedTriangles,
            outTriangleLabels,
            outTriangleConfidence,
            error
        ))
    {
        return false;
    }

    const bool automaticStepInProgress =
        settings.automatic
        && settings.stepByStep
        && static_cast<int>(seedTriangles.size()) < std::max(settings.automaticSeedCount, 1);

    if (settings.automatic && settings.automaticMode == AutomaticSegmentationMode::Fine
        && settings.mergeFineSegments && !automaticStepInProgress)
    {
        const uint32_t mergeTargetSegmentCount =
            static_cast<uint32_t>(std::max(settings.mergeTargetSegmentCount, 1));
        const double mergeCostLimit = settings.forceMergeTargetSegmentCount
            ? std::numeric_limits<double>::infinity()
            : static_cast<double>(settings.mergeCostThreshold);
        if (!segmesh::mergeSegmentsByBoundaryCost(
                mesh,
                settings.model,
                mergeTargetSegmentCount,
                mergeCostLimit,
                outTriangleLabels,
                error
            ))
        {
            return false;
        }
    }

    if (settings.automatic && settings.automaticMode == AutomaticSegmentationMode::Fine
        && settings.cleanupSmallFineSegments && !automaticStepInProgress)
    {
        const uint32_t minTriangleCount =
            static_cast<uint32_t>(std::max(settings.minFineSegmentTriangles, 2));
        if (!segmesh::mergeSmallSegmentsByTriangleCount(
                mesh,
                settings.model,
                minTriangleCount,
                outTriangleLabels,
                error
            ))
        {
            return false;
        }
    }

    for (const ManualSegmentMerge& merge : manualSegmentMerges)
    {
        if (!segmesh::mergeAdjacentSegmentsByTriangles(
                mesh,
                merge.sourceTriangle,
                merge.targetTriangle,
                outTriangleLabels,
                error
            ))
        {
            return false;
        }
    }

    outSegmentCount = 0;
    for (const uint32_t label : outTriangleLabels)
    {
        outSegmentCount = std::max(outSegmentCount, label + 1);
    }

    return true;
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
