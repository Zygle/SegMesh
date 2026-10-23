#include "segmentation_session.h"
#include "segmentation.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace segmesh
{
const std::vector<uint32_t>& SegmentationSession::activeSeeds(bool automatic) const
{
    return automatic ? automaticSeedTriangles : manualSeedTriangles;
}

const std::vector<uint32_t>& SegmentationSession::displayedTriangleLabels() const
{
    return displayedTriangleLabels_;
}

void SegmentationSession::clearDisplayedPreview()
{
    displayedTriangleLabels_.clear();
    cancelMergeSelection();
}

void SegmentationSession::publishPreviewLabels(std::vector<uint32_t> triangleLabels)
{
    displayedTriangleLabels_ = std::move(triangleLabels);
    cancelMergeSelection();
}

std::size_t SegmentationSession::manualMergeCount() const
{
    return manualSegmentMerges_.size();
}

std::optional<uint32_t> SegmentationSession::selectedMergeLabel() const
{
    if (!selectedMergeTriangle_.has_value()
        || *selectedMergeTriangle_ >= displayedTriangleLabels_.size())
    {
        return std::nullopt;
    }
    return displayedTriangleLabels_[*selectedMergeTriangle_];
}

MergePickResult SegmentationSession::handleMergePick(
    const CpuMesh& mesh,
    uint32_t triangle,
    std::string& error
)
{
    error.clear();
    const std::size_t faceCount = mesh.indices.size() / 3;
    if (faceCount == 0 || displayedTriangleLabels_.size() != faceCount)
    {
        error = "No segmentation preview is available to merge.";
        return MergePickResult::Failed;
    }
    if (triangle >= faceCount
        || (selectedMergeTriangle_.has_value() && *selectedMergeTriangle_ >= faceCount))
    {
        error = "Selected triangle index is out of range.";
        return MergePickResult::Failed;
    }

    if (!selectedMergeTriangle_.has_value())
    {
        selectedMergeTriangle_ = triangle;
        return MergePickResult::Selected;
    }
    if (displayedTriangleLabels_[*selectedMergeTriangle_] == displayedTriangleLabels_[triangle])
    {
        cancelMergeSelection();
        return MergePickResult::Cancelled;
    }

    std::vector<uint32_t> validationLabels = displayedTriangleLabels_;
    if (!mergeAdjacentSegmentsByTriangles(
            mesh, *selectedMergeTriangle_, triangle, validationLabels, error
        ))
    {
        return MergePickResult::Failed;
    }

    manualSegmentMerges_.push_back({*selectedMergeTriangle_, triangle});
    cancelMergeSelection();
    return MergePickResult::Merged;
}

void SegmentationSession::cancelMergeSelection()
{
    selectedMergeTriangle_.reset();
}

bool SegmentationSession::undoManualMerge()
{
    if (manualSegmentMerges_.empty())
    {
        return false;
    }
    manualSegmentMerges_.pop_back();
    cancelMergeSelection();
    return true;
}

void SegmentationSession::clearManualMerges()
{
    manualSegmentMerges_.clear();
    cancelMergeSelection();
}

int SegmentationSession::automaticStepSeedCount() const
{
    return automaticStepSeedCount_;
}

void SegmentationSession::updateAutomaticStep(
    const SegmentationSettings& settings,
    bool previousStepByStep,
    AutomaticStepAction action
)
{
    if (!settings.automatic || !settings.stepByStep)
    {
        return;
    }

    automaticStepSeedCount_ = std::max(automaticStepSeedCount_, 1);
    if (!previousStepByStep || action == AutomaticStepAction::Reset)
    {
        automaticStepSeedCount_ = 1;
    }
    else if (action == AutomaticStepAction::Finish)
    {
        automaticStepSeedCount_ = automaticSeedPlanTriangles.empty()
            ? std::numeric_limits<int>::max()
            : static_cast<int>(automaticSeedPlanTriangles.size());
    }
    else if (action == AutomaticStepAction::Next)
    {
        // Do not signal a step change after the generated plan is exhausted:
        // that would unnecessarily refresh the preview and clear manual merges.
        if (!automaticSeedPlanTriangles.empty()
            && static_cast<std::size_t>(automaticStepSeedCount_) >= automaticSeedPlanTriangles.size())
        {
            return;
        }
        if (automaticStepSeedCount_ < std::numeric_limits<int>::max())
        {
            ++automaticStepSeedCount_;
        }
    }
}

void SegmentationSession::resetForMesh()
{
    // Keep the requested step count across mesh changes; prefix selection clamps it.
    manualSeedTriangles.clear();
    automaticSeedPlanTriangles.clear();
    automaticSeedTriangles.clear();
    clearManualMerges();
    clearDisplayedPreview();
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
    std::string& error
)
{
    const uint32_t planSeedCount = static_cast<uint32_t>(automaticSeedPlanTriangles.size());
    if (planSeedCount == 0)
    {
        automaticStepSeedCount_ = 0;
        automaticSeedTriangles.clear();
        error = "Automatic segmentation has no generated seed plan.";
        return false;
    }

    uint32_t activeSeedCount = planSeedCount;
    if (settings.stepByStep)
    {
        const uint32_t requestedSeedCount = static_cast<uint32_t>(std::max(automaticStepSeedCount_, 1));
        activeSeedCount = std::min(requestedSeedCount, planSeedCount);
    }

    automaticStepSeedCount_ = static_cast<int>(activeSeedCount);
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

    // Seed generation can produce fewer seeds than requested. The full generated
    // plan completes stepping and enables the final Fine merge/cleanup passes.
    const bool automaticStepInProgress =
        settings.automatic
        && settings.stepByStep
        && seedTriangles.size() < automaticSeedPlanTriangles.size();

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

    for (const ManualSegmentMerge& merge : manualSegmentMerges_)
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
