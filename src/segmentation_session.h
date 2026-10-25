#pragma once

#include "border_curves.h"
#include "segmentation_settings.h"

#include <cstddef>
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

enum class AutomaticStepAction
{
    None,
    Next,
    Reset,
    Finish,
};

enum class MergePickResult
{
    Selected,
    Cancelled,
    Merged,
    Failed,
};

class SegmentationSession
{
public:
    std::vector<uint32_t> manualSeedTriangles;
    std::vector<uint32_t> automaticSeedPlanTriangles;
    std::vector<uint32_t> automaticSeedTriangles;

    const std::vector<uint32_t>& activeSeeds(bool automatic) const;
    const std::vector<uint32_t>& displayedTriangleLabels() const;
    uint32_t displayedRegionCount() const;
    uint32_t displayedTinyRegionCount() const;
    void clearDisplayedPreview();
    // Call only after the triangle groups and border overlay have been uploaded.
    void publishPreviewLabels(std::vector<uint32_t> triangleLabels);
    std::size_t manualMergeCount() const;
    std::optional<uint32_t> selectedMergeLabel() const;
    MergePickResult handleMergePick(const CpuMesh& mesh, uint32_t triangle, std::string& error);
    void cancelMergeSelection();
    bool undoManualMerge();
    void clearManualMerges();
    int automaticStepSeedCount() const;
    void updateAutomaticStep(
        const SegmentationSettings& settings,
        bool previousStepByStep,
        AutomaticStepAction action
    );
    void resetForMesh();
    bool buildBorderOverlay(
        const CpuMesh& mesh,
        std::span<const uint32_t> triangleLabels,
        const BorderCurveSettings& settings,
        BorderCurveOverlay& outOverlay,
        std::string& error
    );
    bool buildAutomaticSeedPlan(
        const CpuMesh& mesh,
        const SegmentationSettings& settings,
        std::string& error
    );
    bool selectAutomaticSeedPrefix(
        const SegmentationSettings& settings,
        std::string& error
    );
    bool buildPreviewLabels(
        const CpuMesh& mesh,
        const SegmentationSettings& settings,
        const std::vector<uint32_t>& seedTriangles,
        std::vector<uint32_t>& outTriangleLabels,
        std::vector<float>& outTriangleConfidence,
        uint32_t& outSegmentCount,
        std::string& error
    );
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
    int automaticStepSeedCount_ = 1;
    uint32_t minimumAutomaticSeedCount_ = 1;
    std::vector<uint32_t> displayedTriangleLabels_;
    uint32_t displayedRegionCount_ = 0;
    uint32_t displayedTinyRegionCount_ = 0;
    std::vector<ManualSegmentMerge> manualSegmentMerges_;
    std::optional<uint32_t> selectedMergeTriangle_;
    RandomWalkLabelCache randomWalkLabelCache_;
    BorderCurveCache borderCurveCache_;
};
}
