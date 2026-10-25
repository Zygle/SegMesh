#include "seed_selection.h"
#include "face_graph.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace
{
using segmesh::detail::edgeDifferenceForMode;

constexpr double kFineSeedMinImportance = 0.25;
constexpr double kFineSeedMaxImportance = 6.0;
constexpr double kFineSeedSpacingExponent = 1.25;

enum class FaceDistanceMetric
{
    GeodesicLength,
    FeatureWeightedGeodesic,
};

double squaredDistance(const segmesh::Float3& a, const Eigen::Vector3d& b)
{
    const double dx = static_cast<double>(a.x) - b.x();
    const double dy = static_cast<double>(a.y) - b.y();
    const double dz = static_cast<double>(a.z) - b.z();
    return dx * dx + dy * dy + dz * dz;
}

double centroidDistance(const segmesh::Float3& a, const segmesh::Float3& b)
{
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    const double dz = static_cast<double>(a.z) - static_cast<double>(b.z);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double automaticSeedEdgeCost(
    const segmesh::CpuMesh& mesh,
    segmesh::SegmentationModelType segmentationModelType,
    uint32_t faceIndex,
    std::size_t edgeSlot,
    uint32_t neighborIndex,
    FaceDistanceMetric metric
)
{
    if (metric == FaceDistanceMetric::GeodesicLength)
    {
        const double geodesicStep = centroidDistance(
            mesh.faceCentroids[static_cast<std::size_t>(faceIndex)],
            mesh.faceCentroids[static_cast<std::size_t>(neighborIndex)]
        );
        return std::max(geodesicStep, 1.0e-12);
    }

    const double difference = edgeDifferenceForMode(mesh, segmentationModelType, faceIndex, edgeSlot, neighborIndex);
    const double geodesicStep = centroidDistance(
        mesh.faceCentroids[static_cast<std::size_t>(faceIndex)],
        mesh.faceCentroids[static_cast<std::size_t>(neighborIndex)]
    );
    return std::max(geodesicStep * (1.0 + std::max(difference, 0.0)), 1.0e-12);
}

void collectConnectedFaceComponents(
    const segmesh::CpuMesh& mesh,
    std::vector<int32_t>& outComponentIds,
    std::vector<std::vector<uint32_t> >& outComponents
)
{
    const uint32_t faceCount = static_cast<uint32_t>(mesh.faceAdjacency.size());
    outComponentIds.assign(faceCount, -1);
    outComponents.clear();

    std::vector<uint32_t> stack;
    for (uint32_t startFace = 0; startFace < faceCount; ++startFace)
    {
        if (outComponentIds[static_cast<std::size_t>(startFace)] >= 0)
        {
            continue;
        }

        const int32_t componentId = static_cast<int32_t>(outComponents.size());
        outComponents.emplace_back();
        stack.clear();
        stack.push_back(startFace);
        outComponentIds[static_cast<std::size_t>(startFace)] = componentId;

        while (!stack.empty())
        {
            const uint32_t faceIndex = stack.back();
            stack.pop_back();
            outComponents.back().push_back(faceIndex);

            const segmesh::FaceAdjacency& adjacency = mesh.faceAdjacency[static_cast<std::size_t>(faceIndex)];
            for (const int32_t neighborIndex : adjacency.neighbors)
            {
                if (neighborIndex < 0)
                {
                    continue;
                }

                const std::size_t neighborOffset = static_cast<std::size_t>(neighborIndex);
                if (outComponentIds[neighborOffset] >= 0)
                {
                    continue;
                }

                outComponentIds[neighborOffset] = componentId;
                stack.push_back(static_cast<uint32_t>(neighborIndex));
            }
        }
    }
}

void runFaceDijkstra(
    const segmesh::CpuMesh& mesh,
    segmesh::SegmentationModelType segmentationModelType,
    FaceDistanceMetric metric,
    uint32_t sourceFace,
    int32_t componentId,
    const std::vector<int32_t>& componentIds,
    std::vector<double>& outDistances
)
{
    const double infinity = std::numeric_limits<double>::infinity();
    outDistances.assign(mesh.faceAdjacency.size(), infinity);

    using QueueEntry = std::pair<double, uint32_t>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry> > queue;
    outDistances[static_cast<std::size_t>(sourceFace)] = 0.0;
    queue.emplace(0.0, sourceFace);

    while (!queue.empty())
    {
        const QueueEntry current = queue.top();
        queue.pop();

        const double currentDistance = current.first;
        const uint32_t faceIndex = current.second;
        if (currentDistance > outDistances[static_cast<std::size_t>(faceIndex)])
        {
            continue;
        }

        const segmesh::FaceAdjacency& adjacency = mesh.faceAdjacency[static_cast<std::size_t>(faceIndex)];
        for (std::size_t edgeSlot = 0; edgeSlot < adjacency.neighbors.size(); ++edgeSlot)
        {
            const int32_t neighborIndex = adjacency.neighbors[edgeSlot];
            if (neighborIndex < 0 || componentIds[static_cast<std::size_t>(neighborIndex)] != componentId)
            {
                continue;
            }

            const uint32_t neighborFace = static_cast<uint32_t>(neighborIndex);
            const double candidateDistance = currentDistance
                + automaticSeedEdgeCost(mesh, segmentationModelType, faceIndex, edgeSlot, neighborFace, metric);
            if (candidateDistance >= outDistances[static_cast<std::size_t>(neighborFace)])
            {
                continue;
            }

            outDistances[static_cast<std::size_t>(neighborFace)] = candidateDistance;
            queue.emplace(candidateDistance, neighborFace);
        }
    }
}

uint32_t faceClosestToComponentCentroid(const segmesh::CpuMesh& mesh, const std::vector<uint32_t>& componentFaces)
{
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const uint32_t faceIndex : componentFaces)
    {
        const segmesh::Float3& faceCentroid = mesh.faceCentroids[static_cast<std::size_t>(faceIndex)];
        centroid += Eigen::Vector3d(faceCentroid.x, faceCentroid.y, faceCentroid.z);
    }
    centroid /= static_cast<double>(componentFaces.size());

    uint32_t bestFace = componentFaces.front();
    double bestDistance = std::numeric_limits<double>::infinity();
    for (const uint32_t faceIndex : componentFaces)
    {
        const double distance = squaredDistance(mesh.faceCentroids[static_cast<std::size_t>(faceIndex)], centroid);
        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestFace = faceIndex;
        }
    }

    return bestFace;
}

uint32_t farthestReachableFace(const std::vector<uint32_t>& componentFaces, const std::vector<double>& distances)
{
    uint32_t bestFace = componentFaces.front();
    double bestDistance = -1.0;
    for (const uint32_t faceIndex : componentFaces)
    {
        const double distance = distances[static_cast<std::size_t>(faceIndex)];
        if (!std::isfinite(distance) || distance <= bestDistance)
        {
            continue;
        }

        bestDistance = distance;
        bestFace = faceIndex;
    }

    return bestFace;
}

std::vector<double> computeFineSeedImportance(
    const segmesh::CpuMesh& mesh,
    segmesh::SegmentationModelType segmentationModelType
)
{
    const uint32_t faceCount = static_cast<uint32_t>(mesh.faceCentroids.size());
    std::vector<double> importance(faceCount, 1.0);
    if (faceCount == 0)
    {
        return importance;
    }

    Eigen::Vector3d meshCentroid = Eigen::Vector3d::Zero();
    for (const segmesh::Float3& faceCentroid : mesh.faceCentroids)
    {
        meshCentroid += Eigen::Vector3d(faceCentroid.x, faceCentroid.y, faceCentroid.z);
    }
    meshCentroid /= static_cast<double>(faceCount);

    double averageArea = 0.0;
    if (!mesh.faceAreas.empty())
    {
        for (const float faceArea : mesh.faceAreas)
        {
            averageArea += std::max(static_cast<double>(faceArea), 1.0e-12);
        }
        averageArea /= static_cast<double>(mesh.faceAreas.size());
    }
    averageArea = std::max(averageArea, 1.0e-12);

    double maxRadius = 0.0;
    for (const segmesh::Float3& faceCentroid : mesh.faceCentroids)
    {
        maxRadius = std::max(maxRadius, std::sqrt(squaredDistance(faceCentroid, meshCentroid)));
    }
    maxRadius = std::max(maxRadius, 1.0e-6);

    for (uint32_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
    {
        const segmesh::FaceAdjacency& adjacency = mesh.faceAdjacency[static_cast<std::size_t>(faceIndex)];

        double variationSum = 0.0;
        double gaussianVariationSum = 0.0;
        double meanVariationSum = 0.0;
        double concavitySum = 0.0;
        double edgeLengthSum = 0.0;
        uint32_t neighborCount = 0;
        for (std::size_t edgeSlot = 0; edgeSlot < adjacency.neighbors.size(); ++edgeSlot)
        {
            const int32_t neighborIndex = adjacency.neighbors[edgeSlot];
            if (neighborIndex < 0)
            {
                continue;
            }

            const double normalDifference = segmesh::detail::faceNormalDifference(
                mesh,
                faceIndex,
                static_cast<uint32_t>(neighborIndex)
            );
            variationSum += normalDifference;
            if (segmentationModelType == segmesh::SegmentationModelType::Engineering)
            {
                gaussianVariationSum += std::abs(
                    static_cast<double>(mesh.faceGaussianCurvatures[faceIndex])
                    - static_cast<double>(mesh.faceGaussianCurvatures[neighborIndex])
                );
                meanVariationSum += std::abs(
                    static_cast<double>(mesh.faceMeanCurvatures[faceIndex])
                    - static_cast<double>(mesh.faceMeanCurvatures[neighborIndex])
                );
            }
            concavitySum += static_cast<double>(adjacency.concavityScales[edgeSlot]);
            edgeLengthSum += static_cast<double>(adjacency.edgeLengths[edgeSlot]);
            ++neighborCount;
        }

        const double normalVariation = neighborCount > 0 ? variationSum / static_cast<double>(neighborCount) : 0.0;
        const double concavityBias = neighborCount > 0 ? concavitySum / static_cast<double>(neighborCount) : 0.2;
        const double averageEdgeLength =
            neighborCount > 0 ? edgeLengthSum / static_cast<double>(neighborCount) : 0.0;
        const double radialDistance =
            std::sqrt(squaredDistance(mesh.faceCentroids[static_cast<std::size_t>(faceIndex)], meshCentroid)) / maxRadius;
        if (segmentationModelType == segmesh::SegmentationModelType::Engineering)
        {
            // Compare local changes with global changes, not absolute curvature
            // with its derivative. Constant-curvature patches should not saturate.
            const double divisor = static_cast<double>(std::max(neighborCount, 1u));
            const double gaussianVariation = gaussianVariationSum / divisor
                / std::max(static_cast<double>(mesh.averageGaussianDifference), 1.0e-12);
            const double meanVariation = meanVariationSum / divisor
                / std::max(static_cast<double>(mesh.averageMeanDifference), 1.0e-12);
            const double scaledNormalVariation = normalVariation
                / std::max(static_cast<double>(mesh.averageEngineeringNormalDifference), 1.0e-12);
            const auto boundedFeature = [](double value) { return value / (1.0 + value); };
            // Keep feature preference secondary to surface coverage (importance 1..1.75).
            importance[faceIndex] = 1.0 + 0.25 * (
                boundedFeature(scaledNormalVariation)
                + boundedFeature(gaussianVariation)
                + boundedFeature(meanVariation)
            );
            continue;
        }

        const double faceArea =
            faceIndex < mesh.faceAreas.size() ? std::max(static_cast<double>(mesh.faceAreas[faceIndex]), 1.0e-12) : averageArea;
        const double areaWeight = std::sqrt(faceArea / averageArea);

        // Sec. 4.2.1: bias dense seeds toward protrusions and locally featured regions.
        const double featureBias =
            1.0 + 3.0 * normalVariation + 0.75 * radialDistance + 0.5 * std::max(concavityBias - 0.2, 0.0);
        const double spacingBias = 0.5 + 0.5 * std::clamp(areaWeight * (1.0 + averageEdgeLength), 0.0, 2.0);
        importance[static_cast<std::size_t>(faceIndex)] =
            std::clamp(featureBias * spacingBias, kFineSeedMinImportance, kFineSeedMaxImportance);
    }

    return importance;
}

uint32_t mostImportantFace(
    const std::vector<uint32_t>& componentFaces,
    const std::vector<double>& importance,
    const std::vector<bool>& isSeed
)
{
    uint32_t bestFace = componentFaces.front();
    double bestImportance = -1.0;
    for (const uint32_t faceIndex : componentFaces)
    {
        if (isSeed[static_cast<std::size_t>(faceIndex)])
        {
            continue;
        }

        const double faceImportance = importance[static_cast<std::size_t>(faceIndex)];
        if (faceImportance > bestImportance)
        {
            bestImportance = faceImportance;
            bestFace = faceIndex;
        }
    }

    return bestFace;
}

}

namespace segmesh
{
bool selectAutomaticSeedsCoarse(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t approximateSeedCount,
    std::vector<uint32_t>& outSeedTriangles,
    std::string& error,
    uint32_t* outMinimumSeedCount
)
{
    outSeedTriangles.clear();
    error.clear();
    if (outMinimumSeedCount != nullptr)
    {
        *outMinimumSeedCount = 0;
    }

    const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (faceCount == 0)
    {
        error = "Cannot auto-segment an empty mesh.";
        return false;
    }

    if (approximateSeedCount == 0)
    {
        error = "Automatic segmentation requires at least one target seed.";
        return false;
    }

    if (mesh.faceCentroids.size() != faceCount || mesh.faceNormals.size() != faceCount
        || mesh.faceAdjacency.size() != faceCount)
    {
        error = "Mesh topology data is incomplete. Reload the mesh before auto-segmenting.";
        return false;
    }

    if (!detail::validateFaceGraphData(mesh, segmentationModelType, error))
    {
        return false;
    }

    std::vector<int32_t> componentIds;
    std::vector<std::vector<uint32_t> > components;
    collectConnectedFaceComponents(mesh, componentIds, components);
    if (components.empty())
    {
        error = "Automatic segmentation could not find any face components.";
        return false;
    }

    const uint32_t targetSeedCount =
        std::max<uint32_t>(approximateSeedCount, static_cast<uint32_t>(components.size()));
    outSeedTriangles.reserve(targetSeedCount);

    std::vector<bool> isSeed(faceCount, false);
    std::vector<double> minDistances(faceCount, std::numeric_limits<double>::infinity());
    std::vector<double> distances(faceCount, std::numeric_limits<double>::infinity());

    const auto addSeedAndUpdateDistances = [&](uint32_t seedFace) -> bool
    {
        if (seedFace >= faceCount || isSeed[static_cast<std::size_t>(seedFace)])
        {
            return false;
        }

        outSeedTriangles.push_back(seedFace);
        isSeed[static_cast<std::size_t>(seedFace)] = true;

        const int32_t componentId = componentIds[static_cast<std::size_t>(seedFace)];
        const std::vector<uint32_t>& componentFaces = components[static_cast<std::size_t>(componentId)];
        runFaceDijkstra(
            mesh,
            segmentationModelType,
            FaceDistanceMetric::GeodesicLength,
            seedFace,
            componentId,
            componentIds,
            distances
        );
        for (const uint32_t faceIndex : componentFaces)
        {
            const std::size_t faceOffset = static_cast<std::size_t>(faceIndex);
            minDistances[faceOffset] = std::min(minDistances[faceOffset], distances[faceOffset]);
        }

        return true;
    };

    const auto componentAlreadyHasSeed = [&](const std::vector<uint32_t>& componentFaces) -> bool
    {
        for (const uint32_t faceIndex : componentFaces)
        {
            if (isSeed[static_cast<std::size_t>(faceIndex)])
            {
                return true;
            }
        }

        return false;
    };

    for (std::size_t componentIndex = 0; componentIndex < components.size()
         && outSeedTriangles.size() < targetSeedCount;
         ++componentIndex)
    {
        const std::vector<uint32_t>& componentFaces = components[componentIndex];
        if (componentAlreadyHasSeed(componentFaces))
        {
            continue;
        }

        // Paper Sec. 4.1: start from the face nearest the component centroid, then take the farthest face as s1.
        const uint32_t centroidFace = faceClosestToComponentCentroid(mesh, componentFaces);
        runFaceDijkstra(
            mesh,
            segmentationModelType,
            FaceDistanceMetric::GeodesicLength,
            centroidFace,
            static_cast<int32_t>(componentIndex),
            componentIds,
            distances
        );

        const uint32_t firstSeed = farthestReachableFace(componentFaces, distances);
        addSeedAndUpdateDistances(firstSeed);
    }

    while (outSeedTriangles.size() < targetSeedCount)
    {
        uint32_t nextSeed = 0;
        double nextSeedDistance = -1.0;

        for (uint32_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
        {
            if (isSeed[static_cast<std::size_t>(faceIndex)])
            {
                continue;
            }

            const double distance = minDistances[static_cast<std::size_t>(faceIndex)];
            if (!std::isfinite(distance) || distance <= nextSeedDistance)
            {
                continue;
            }

            nextSeedDistance = distance;
            nextSeed = faceIndex;
        }

        if (nextSeedDistance <= 1.0e-6)
        {
            break;
        }

        // Eq. (13): pick the face with maximal distance to its nearest existing seed.
        addSeedAndUpdateDistances(nextSeed);
    }

    if (outSeedTriangles.empty())
    {
        error = "Automatic segmentation did not produce any seed triangles.";
        return false;
    }

    if (outMinimumSeedCount != nullptr)
    {
        *outMinimumSeedCount = static_cast<uint32_t>(components.size());
    }
    return true;
}

bool selectAutomaticSeedsFine(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    uint32_t approximateSeedCount,
    std::vector<uint32_t>& outSeedTriangles,
    std::string& error,
    uint32_t* outMinimumSeedCount
)
{
    outSeedTriangles.clear();
    error.clear();
    if (outMinimumSeedCount != nullptr)
    {
        *outMinimumSeedCount = 0;
    }

    const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (faceCount == 0)
    {
        error = "Cannot auto-segment an empty mesh.";
        return false;
    }

    if (approximateSeedCount == 0)
    {
        error = "Automatic segmentation requires at least one target seed.";
        return false;
    }

    if (mesh.faceCentroids.size() != faceCount || mesh.faceNormals.size() != faceCount
        || mesh.faceAdjacency.size() != faceCount)
    {
        error = "Mesh topology data is incomplete. Reload the mesh before auto-segmenting.";
        return false;
    }

    if (!detail::validateFaceGraphData(mesh, segmentationModelType, error))
    {
        return false;
    }

    std::vector<int32_t> componentIds;
    std::vector<std::vector<uint32_t> > components;
    collectConnectedFaceComponents(mesh, componentIds, components);
    if (components.empty())
    {
        error = "Automatic segmentation could not find any face components.";
        return false;
    }

    const uint32_t targetSeedCount = std::min<uint32_t>(
        faceCount,
        std::max<uint32_t>(approximateSeedCount, static_cast<uint32_t>(components.size()))
    );
    outSeedTriangles.reserve(targetSeedCount);

    const std::vector<double> importance = computeFineSeedImportance(mesh, segmentationModelType);
    const bool engineering = segmentationModelType == SegmentationModelType::Engineering;
    // Engineering transition costs may be very large at sharp edges. Using them
    // as sampling distances can spend the seed budget on tiny isolated patches.
    const FaceDistanceMetric samplingMetric = engineering
        ? FaceDistanceMetric::GeodesicLength
        : FaceDistanceMetric::FeatureWeightedGeodesic;
    std::vector<bool> isSeed(faceCount, false);
    std::vector<double> minDistances(faceCount, std::numeric_limits<double>::infinity());
    std::vector<double> distances(faceCount, std::numeric_limits<double>::infinity());

    // Sec. 4.2.1: seed each component first, then densify with many feature-biased samples.
    for (std::size_t componentIndex = 0; componentIndex < components.size(); ++componentIndex)
    {
        const std::vector<uint32_t>& componentFaces = components[componentIndex];
        const uint32_t initialSeed = engineering
            ? faceClosestToComponentCentroid(mesh, componentFaces)
            : mostImportantFace(componentFaces, importance, isSeed);
        outSeedTriangles.push_back(initialSeed);
        isSeed[static_cast<std::size_t>(initialSeed)] = true;

        runFaceDijkstra(
            mesh,
            segmentationModelType,
            samplingMetric,
            initialSeed,
            static_cast<int32_t>(componentIndex),
            componentIds,
            distances
        );
        for (const uint32_t faceIndex : componentFaces)
        {
            minDistances[static_cast<std::size_t>(faceIndex)] = distances[static_cast<std::size_t>(faceIndex)];
        }
    }

    while (outSeedTriangles.size() < targetSeedCount)
    {
        uint32_t nextSeed = 0;
        double nextScore = -1.0;

        for (uint32_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
        {
            if (isSeed[static_cast<std::size_t>(faceIndex)])
            {
                continue;
            }

            const double distance = minDistances[static_cast<std::size_t>(faceIndex)];
            if (!std::isfinite(distance))
            {
                continue;
            }

            // Sec. 4.2.1 approximation: keep feature bias, but make spacing dominate tiny/noisy features.
            const double featureWeight = std::sqrt(importance[static_cast<std::size_t>(faceIndex)]);
            const double spacingWeight = std::pow(std::max(distance, 0.0), kFineSeedSpacingExponent);
            const double score = spacingWeight * featureWeight;
            if (score > nextScore)
            {
                nextScore = score;
                nextSeed = faceIndex;
            }
        }

        if (nextScore <= 1.0e-6)
        {
            break;
        }

        outSeedTriangles.push_back(nextSeed);
        isSeed[static_cast<std::size_t>(nextSeed)] = true;

        const int32_t componentId = componentIds[static_cast<std::size_t>(nextSeed)];
        const std::vector<uint32_t>& componentFaces = components[static_cast<std::size_t>(componentId)];
        runFaceDijkstra(
            mesh,
            segmentationModelType,
            samplingMetric,
            nextSeed,
            componentId,
            componentIds,
            distances
        );
        for (const uint32_t faceIndex : componentFaces)
        {
            const std::size_t faceOffset = static_cast<std::size_t>(faceIndex);
            minDistances[faceOffset] = std::min(minDistances[faceOffset], distances[faceOffset]);
        }
    }

    if (outSeedTriangles.empty())
    {
        error = "Automatic segmentation did not produce any seed triangles.";
        return false;
    }

    if (outMinimumSeedCount != nullptr)
    {
        *outMinimumSeedCount = static_cast<uint32_t>(components.size());
    }
    return true;
}

}
