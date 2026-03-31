#include "segmentation.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

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
double dot(const segmesh::Float3& a, const segmesh::Float3& b)
{
    return static_cast<double>(a.x) * static_cast<double>(b.x)
        + static_cast<double>(a.y) * static_cast<double>(b.y)
        + static_cast<double>(a.z) * static_cast<double>(b.z);
}

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
    uint32_t faceIndex,
    std::size_t edgeSlot,
    uint32_t neighborIndex
)
{
    const double normalDot = std::clamp(
        dot(
            mesh.faceNormals[static_cast<std::size_t>(faceIndex)],
            mesh.faceNormals[static_cast<std::size_t>(neighborIndex)]
        ),
        -1.0,
        1.0
    );
    const double difference =
        static_cast<double>(mesh.faceAdjacency[static_cast<std::size_t>(faceIndex)].concavityScales[edgeSlot])
        * (1.0 - normalDot);
    const double geodesicStep = centroidDistance(
        mesh.faceCentroids[static_cast<std::size_t>(faceIndex)],
        mesh.faceCentroids[static_cast<std::size_t>(neighborIndex)]
    );
    return std::max(geodesicStep * (1.0 + difference), 1.0e-6);
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
            const double candidateDistance =
                currentDistance + automaticSeedEdgeCost(mesh, faceIndex, edgeSlot, neighborFace);
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

std::vector<double> computeFineSeedImportance(const segmesh::CpuMesh& mesh)
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

            const double normalDot = std::clamp(
                dot(
                    mesh.faceNormals[static_cast<std::size_t>(faceIndex)],
                    mesh.faceNormals[static_cast<std::size_t>(neighborIndex)]
                ),
                -1.0,
                1.0
            );
            variationSum += 1.0 - normalDot;
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

        const double faceArea =
            faceIndex < mesh.faceAreas.size() ? std::max(static_cast<double>(mesh.faceAreas[faceIndex]), 1.0e-12) : averageArea;
        const double areaWeight = std::sqrt(faceArea / averageArea);

        // Sec. 4.2.1: bias dense seeds toward protrusions and locally featured regions.
        const double featureBias =
            1.0 + 3.0 * normalVariation + 0.75 * radialDistance + 0.5 * std::max(concavityBias - 0.2, 0.0);
        const double spacingBias = 0.5 + 0.5 * std::clamp(areaWeight * (1.0 + averageEdgeLength), 0.0, 2.0);
        importance[static_cast<std::size_t>(faceIndex)] = std::max(featureBias * spacingBias, 1.0e-6);
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

bool componentHasSeed(
    const segmesh::CpuMesh& mesh,
    uint32_t startFace,
    const std::vector<bool>& isSeed,
    std::vector<bool>& visited
)
{
    // Traverse one connected face component and check whether it contains any seed.
    std::vector<uint32_t> stack;
    stack.push_back(startFace);
    visited[static_cast<std::size_t>(startFace)] = true;

    bool foundSeed = false;
    while (!stack.empty())
    {
        const uint32_t faceIndex = stack.back();
        stack.pop_back();
        foundSeed = foundSeed || isSeed[static_cast<std::size_t>(faceIndex)];

        const segmesh::FaceAdjacency& adjacency = mesh.faceAdjacency[static_cast<std::size_t>(faceIndex)];
        for (const int32_t neighborIndex : adjacency.neighbors)
        {
            if (neighborIndex < 0)
            {
                continue;
            }

            const std::size_t neighborOffset = static_cast<std::size_t>(neighborIndex);
            if (visited[neighborOffset])
            {
                continue;
            }

            visited[neighborOffset] = true;
            stack.push_back(static_cast<uint32_t>(neighborIndex));
        }
    }

    return foundSeed;
}
}

namespace segmesh
{
bool selectAutomaticSeedsCoarse(
    const CpuMesh& mesh,
    uint32_t approximateSeedCount,
    std::vector<uint32_t>& outSeedTriangles,
    std::string& error
)
{
    outSeedTriangles.clear();
    error.clear();

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

    for (std::size_t componentIndex = 0; componentIndex < components.size(); ++componentIndex)
    {
        const std::vector<uint32_t>& componentFaces = components[componentIndex];
        // Paper Sec. 4.1: start from the face nearest the component centroid, then take the farthest face as s1.
        const uint32_t centroidFace = faceClosestToComponentCentroid(mesh, componentFaces);
        runFaceDijkstra(mesh, centroidFace, static_cast<int32_t>(componentIndex), componentIds, distances);

        const uint32_t firstSeed = farthestReachableFace(componentFaces, distances);
        outSeedTriangles.push_back(firstSeed);
        isSeed[static_cast<std::size_t>(firstSeed)] = true;

        runFaceDijkstra(mesh, firstSeed, static_cast<int32_t>(componentIndex), componentIds, distances);
        for (const uint32_t faceIndex : componentFaces)
        {
            minDistances[static_cast<std::size_t>(faceIndex)] = distances[static_cast<std::size_t>(faceIndex)];
        }
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
        outSeedTriangles.push_back(nextSeed);
        isSeed[static_cast<std::size_t>(nextSeed)] = true;

        const int32_t componentId = componentIds[static_cast<std::size_t>(nextSeed)];
        const std::vector<uint32_t>& componentFaces = components[static_cast<std::size_t>(componentId)];
        runFaceDijkstra(mesh, nextSeed, componentId, componentIds, distances);
        for (const uint32_t faceIndex : componentFaces)
        {
            // Keep min_i D(f_k, s_i) up to date for the next Eq. (13) selection step.
            const std::size_t faceOffset = static_cast<std::size_t>(faceIndex);
            minDistances[faceOffset] = std::min(minDistances[faceOffset], distances[faceOffset]);
        }
    }

    if (outSeedTriangles.empty())
    {
        error = "Automatic segmentation did not produce any seed triangles.";
        return false;
    }

    return true;
}

bool selectAutomaticSeedsFine(
    const CpuMesh& mesh,
    uint32_t approximateSeedCount,
    std::vector<uint32_t>& outSeedTriangles,
    std::string& error
)
{
    outSeedTriangles.clear();
    error.clear();

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

    const std::vector<double> importance = computeFineSeedImportance(mesh);
    std::vector<bool> isSeed(faceCount, false);
    std::vector<double> minDistances(faceCount, std::numeric_limits<double>::infinity());
    std::vector<double> distances(faceCount, std::numeric_limits<double>::infinity());

    // Sec. 4.2.1: seed each component first, then densify with many feature-biased samples.
    for (std::size_t componentIndex = 0; componentIndex < components.size(); ++componentIndex)
    {
        const std::vector<uint32_t>& componentFaces = components[componentIndex];
        const uint32_t initialSeed = mostImportantFace(componentFaces, importance, isSeed);
        outSeedTriangles.push_back(initialSeed);
        isSeed[static_cast<std::size_t>(initialSeed)] = true;

        runFaceDijkstra(mesh, initialSeed, static_cast<int32_t>(componentIndex), componentIds, distances);
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

            // Sec. 4.2.1: approximate feature-sensitive sampling by combining spacing with local feature importance.
            const double score = distance * importance[static_cast<std::size_t>(faceIndex)];
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
        runFaceDijkstra(mesh, nextSeed, componentId, componentIds, distances);
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

    return true;
}

bool segmentMeshRandomWalk(
    const CpuMesh& mesh,
    const std::vector<uint32_t>& seedTriangles,
    std::vector<uint32_t>& outTriangleLabels,
    std::string& error
)
{
    outTriangleLabels.clear();
    error.clear();

    const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (faceCount == 0)
    {
        error = "Cannot segment an empty mesh.";
        return false;
    }

    if (mesh.faceNormals.size() != faceCount || mesh.faceAdjacency.size() != faceCount)
    {
        error = "Mesh topology data is incomplete. Reload the mesh before segmenting.";
        return false;
    }

    // Each unique seed becomes one segment label and one probability field.
    std::vector<uint32_t> uniqueSeeds;
    uniqueSeeds.reserve(seedTriangles.size());
    std::vector<int32_t> faceLabels(faceCount, -1);

    for (const uint32_t seedFace : seedTriangles)
    {
        if (seedFace >= faceCount)
        {
            error = "Seed triangle index is out of range.";
            return false;
        }

        if (faceLabels[static_cast<std::size_t>(seedFace)] >= 0)
        {
            continue;
        }

        faceLabels[static_cast<std::size_t>(seedFace)] = static_cast<int32_t>(uniqueSeeds.size());
        uniqueSeeds.push_back(seedFace);
    }

    if (uniqueSeeds.empty())
    {
        error = "At least one seed triangle is required.";
        return false;
    }

    std::vector<bool> isSeed(faceCount, false);
    for (const uint32_t seedFace : uniqueSeeds)
    {
        isSeed[static_cast<std::size_t>(seedFace)] = true;
    }

    std::vector<bool> visited(faceCount, false);
    for (uint32_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
    {
        if (visited[static_cast<std::size_t>(faceIndex)])
        {
            continue;
        }

        if (!componentHasSeed(mesh, faceIndex, isSeed, visited))
        {
            error = "Each connected mesh component must contain at least one seed triangle.";
            return false;
        }
    }

    outTriangleLabels.resize(faceCount, 0);
    for (const uint32_t seedFace : uniqueSeeds)
    {
        outTriangleLabels[static_cast<std::size_t>(seedFace)] =
            static_cast<uint32_t>(faceLabels[static_cast<std::size_t>(seedFace)]);
    }

    if (uniqueSeeds.size() == 1)
    {
        return true;
    }


    std::vector<int32_t> rowIndices(faceCount, -1);
    uint32_t unknownCount = 0;
    for (uint32_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
    {
        if (!isSeed[static_cast<std::size_t>(faceIndex)])
        {
            rowIndices[static_cast<std::size_t>(faceIndex)] = static_cast<int32_t>(unknownCount);
            ++unknownCount;
        }
    }

    if (unknownCount == 0)
    {
        return true;
    }

    struct EdgeInfo
    {
        uint32_t face0 = 0;
        uint32_t face1 = 0;
        double edgeLength = 0.0;
        double difference = 0.0;
    };

    std::vector<EdgeInfo> edges;
    edges.reserve(static_cast<std::size_t>(faceCount) * 3 / 2);

    double differenceSum = 0.0;
    // Build one undirected graph edge for each adjacent face pair.
    for (uint32_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
    {
        const FaceAdjacency& adjacency = mesh.faceAdjacency[static_cast<std::size_t>(faceIndex)];
        for (std::size_t edgeSlot = 0; edgeSlot < adjacency.neighbors.size(); ++edgeSlot)
        {
            const int32_t neighborIndex = adjacency.neighbors[edgeSlot];
            if (neighborIndex < 0 || faceIndex >= static_cast<uint32_t>(neighborIndex))
            {
                continue;
            }

            const double normalDot = std::clamp(
                dot(
                    mesh.faceNormals[static_cast<std::size_t>(faceIndex)],
                    mesh.faceNormals[static_cast<std::size_t>(neighborIndex)]
                ),
                -1.0,
                1.0
            );
            const double difference = static_cast<double>(adjacency.concavityScales[edgeSlot]) * (1.0 - normalDot);
            const double edgeLength = std::max(static_cast<double>(adjacency.edgeLengths[edgeSlot]), 1.0e-12);

            //Eq. (1): after normalization affinities become the transition probabilities p{k,i}.
            edges.push_back({faceIndex, static_cast<uint32_t>(neighborIndex), edgeLength, difference});
            differenceSum += difference;
        }
    }

    if (edges.empty())
    {
        error = "The mesh face graph has no adjacency edges to solve over.";
        return false;
    }

    const double averageDifference = std::max(differenceSum / static_cast<double>(edges.size()), 1.0e-12);

    // Assemble a sparse Laplacian and one right-hand side column per seed.
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(unknownCount),
        static_cast<Eigen::Index>(uniqueSeeds.size())
    );
    std::vector<double> diagonal(static_cast<std::size_t>(unknownCount), 0.0);
    std::vector<Eigen::Triplet<double> > triplets;
    triplets.reserve(edges.size() * 4 + diagonal.size());

    for (const EdgeInfo& edge : edges)
    {
        double weight = edge.edgeLength * std::exp(-(edge.difference / averageDifference));
        weight = std::max(weight, 1.0e-12);

        const int32_t row0 = rowIndices[static_cast<std::size_t>(edge.face0)];
        const int32_t row1 = rowIndices[static_cast<std::size_t>(edge.face1)];
        const int32_t label0 = faceLabels[static_cast<std::size_t>(edge.face0)];
        const int32_t label1 = faceLabels[static_cast<std::size_t>(edge.face1)];

        if (row0 >= 0)
        {
            diagonal[static_cast<std::size_t>(row0)] += weight;
            if (label1 >= 0)
            {
                // Seed neighbors are fixed P^l values, so they move to the right-hand side.
                rhs(row0, label1) += weight;
            }
            else
            {
                triplets.emplace_back(row0, row1, -weight);
            }
        }

        if (row1 >= 0)
        {
            diagonal[static_cast<std::size_t>(row1)] += weight;
            if (label0 >= 0)
            {
                rhs(row1, label0) += weight;
            }
            else
            {
                triplets.emplace_back(row1, row0, -weight);
            }
        }
    }

    for (uint32_t row = 0; row < unknownCount; ++row)
    {
        // Eq. (2)
        // sum_j w_ij (x_i - x_j) = 0 for each non-seed face.
        triplets.emplace_back(
            static_cast<Eigen::Index>(row),
            static_cast<Eigen::Index>(row),
            diagonal[static_cast<std::size_t>(row)]
        );
    }

    Eigen::SparseMatrix<double> laplacian(
        static_cast<Eigen::Index>(unknownCount),
        static_cast<Eigen::Index>(unknownCount)
    );
    laplacian.setFromTriplets(triplets.begin(), triplets.end());

    // Factor once, then solve all seed probability fields together.
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
    solver.compute(laplacian);
    if (solver.info() != Eigen::Success)
    {
        error = "Failed to factorize the random-walk sparse system.";
        return false;
    }

    const Eigen::MatrixXd probabilities = solver.solve(rhs);
    if (solver.info() != Eigen::Success)
    {
        error = "Failed to solve the random-walk sparse system.";
        return false;
    }

    for (uint32_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
    {
        if (isSeed[static_cast<std::size_t>(faceIndex)])
        {
            continue;
        }

        const int32_t rowIndex = rowIndices[static_cast<std::size_t>(faceIndex)];
        if (rowIndex < 0)
        {
            error = "Internal segmentation state became inconsistent.";
            return false;
        }

        // Eq. (3) assign the face to the seed with maximal probability.
        Eigen::Index bestLabel = 0;
        probabilities.row(rowIndex).maxCoeff(&bestLabel);
        const double bestValue = probabilities(rowIndex, bestLabel);
        if (!std::isfinite(bestValue))
        {
            error = "Segmentation solve produced a non-finite probability.";
            return false;
        }

        outTriangleLabels[static_cast<std::size_t>(faceIndex)] = static_cast<uint32_t>(bestLabel);
    }

    return true;
}
}
