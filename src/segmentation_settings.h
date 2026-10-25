#pragma once

namespace segmesh
{
enum class SegmentationModelType : int
{
    Graphical = 0,
    Engineering = 1,
};

enum class AutomaticSegmentationMode : int
{
    Coarse = 0,
    Fine = 1,
};

struct SegmentationSettings
{
    bool automatic = false;
    SegmentationModelType model = SegmentationModelType::Graphical;
    AutomaticSegmentationMode automaticMode = AutomaticSegmentationMode::Coarse;
    int coarseSeedCount = 8;
    int fineSeedCount = 60;
    bool stepByStep = false;
    bool mergeFineSegments = true;
    int mergeTargetSegmentCount = 30;
    bool forceMergeTargetSegmentCount = false;
    float mergeCostThreshold = 0.5f;
    bool cleanupSmallFineSegments = false;
    int minFineSegmentTriangles = 24;

    int& automaticSeedCount()
    {
        return automaticMode == AutomaticSegmentationMode::Fine ? fineSeedCount : coarseSeedCount;
    }

    int automaticSeedCount() const
    {
        return automaticMode == AutomaticSegmentationMode::Fine ? fineSeedCount : coarseSeedCount;
    }
};
}
