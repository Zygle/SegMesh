#include "segmentation.h"
#include "face_graph.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{
using segmesh::detail::FaceGraphEdgeInfo;
using segmesh::detail::buildFaceGraphEdges;

constexpr double kProbabilityTolerance = 1.0e-6;
constexpr double kProbabilitySumTolerance = 1.0e-5;
constexpr double kScaledResidualTolerance = 1.0e-8;

bool validateProbabilities(
    const Eigen::SparseMatrix<double>& laplacian,
    const Eigen::MatrixXd& rhs,
    const Eigen::MatrixXd& probabilities,
    std::string& error
)
{
    if (!probabilities.allFinite())
    {
        error = "Segmentation solve produced a non-finite probability.";
        return false;
    }
    for (Eigen::Index row = 0; row < probabilities.rows(); ++row)
    {
        if (probabilities.row(row).minCoeff() < -kProbabilityTolerance
            || probabilities.row(row).maxCoeff() > 1.0 + kProbabilityTolerance)
        {
            error = "Segmentation probabilities are outside zero to one beyond roundoff tolerance.";
            return false;
        }
        if (std::abs(probabilities.row(row).sum() - 1.0) > kProbabilitySumTolerance)
        {
            error = "Segmentation probabilities do not sum to one within tolerance.";
            return false;
        }
    }

    Eigen::VectorXd rowSums = Eigen::VectorXd::Zero(laplacian.rows());
    for (Eigen::Index outer = 0; outer < laplacian.outerSize(); ++outer)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator entry(laplacian, outer); entry; ++entry)
        {
            rowSums(entry.row()) += std::abs(entry.value());
        }
    }
    const double matrixInfinityNorm = rowSums.maxCoeff();
    // Check columns separately to avoid allocating another full face-by-seed matrix.
    for (Eigen::Index column = 0; column < probabilities.cols(); ++column)
    {
        const Eigen::VectorXd residual = laplacian * probabilities.col(column) - rhs.col(column);
        const double scale = matrixInfinityNorm * probabilities.col(column).lpNorm<Eigen::Infinity>()
            + rhs.col(column).lpNorm<Eigen::Infinity>();
        if (!residual.allFinite() || !std::isfinite(scale))
        {
            error = "Segmentation residual calculation produced a non-finite value.";
            return false;
        }
        const double scaledResidual = residual.lpNorm<Eigen::Infinity>()
            / std::max(scale, std::numeric_limits<double>::min());
        if (!std::isfinite(scaledResidual) || scaledResidual > kScaledResidualTolerance)
        {
            error = "Segmentation solve residual exceeds the numerical tolerance.";
            return false;
        }
    }
    return true;
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
bool segmentMeshRandomWalk(
    const CpuMesh& mesh,
    SegmentationModelType segmentationModelType,
    const std::vector<uint32_t>& seedTriangles,
    std::vector<uint32_t>& outTriangleLabels,
    std::vector<float>& outTriangleConfidence,
    std::string& error
)
{
    outTriangleLabels.clear();
    outTriangleConfidence.clear();
    error.clear();

    const uint32_t faceCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    if (faceCount == 0)
    {
        error = "Cannot segment an empty mesh.";
        return false;
    }

    if (!detail::validateFaceGraphData(mesh, segmentationModelType, error))
    {
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
    outTriangleConfidence.resize(faceCount, 1.0f);
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

    std::vector<FaceGraphEdgeInfo> edges;
    double averageDifference = 0.0;
    if (!buildFaceGraphEdges(mesh, segmentationModelType, edges, averageDifference, error))
    {
        return false;
    }

    // Assemble a sparse Laplacian and one right-hand side column per seed.
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(unknownCount),
        static_cast<Eigen::Index>(uniqueSeeds.size())
    );
    std::vector<double> diagonal(static_cast<std::size_t>(unknownCount), 0.0);
    std::vector<Eigen::Triplet<double> > triplets;
    triplets.reserve(edges.size() * 4 + diagonal.size());

    for (const FaceGraphEdgeInfo& edge : edges)
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
        if (!std::isfinite(diagonal[row]) || diagonal[row] <= 0.0)
        {
            error = "Random-walk system contains an invalid diagonal weight.";
            return false;
        }
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

    if (!solver.vectorD().allFinite() || solver.vectorD().minCoeff() <= 0.0)
    {
        error = "Random-walk system is not numerically positive definite.";
        return false;
    }

    const Eigen::MatrixXd probabilities = solver.solve(rhs);
    if (solver.info() != Eigen::Success)
    {
        error = "Failed to solve the random-walk sparse system.";
        return false;
    }
    if (!validateProbabilities(laplacian, rhs, probabilities, error))
    {
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
        outTriangleLabels[static_cast<std::size_t>(faceIndex)] = static_cast<uint32_t>(bestLabel);
        outTriangleConfidence[static_cast<std::size_t>(faceIndex)] =
            static_cast<float>(std::clamp(bestValue, 0.0, 1.0));
    }

    return true;
}
}
