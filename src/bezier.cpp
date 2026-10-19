#include "bezier.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace segmesh
{
namespace
{
constexpr double kCoincidentPointEpsilon = 1.0e-12;
constexpr uint32_t kMaximumFitSamplingDepth = 12;
constexpr uint32_t kMinimumFitSamplingDepth = 5;

Eigen::Vector3d toEigen(const segmesh::Float3& point)
{
    return {
        static_cast<double>(point.x),
        static_cast<double>(point.y),
        static_cast<double>(point.z),
    };
}

segmesh::Float3 toFloat3(const Eigen::Vector3d& point)
{
    return {
        static_cast<float>(point.x()),
        static_cast<float>(point.y()),
        static_cast<float>(point.z()),
    };
}

bool isFinite(const segmesh::Float3& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

double pointDistance(const segmesh::Float3& first, const segmesh::Float3& second)
{
    return (toEigen(first) - toEigen(second)).norm();
}

std::vector<double> chordLengthParameters(std::span<const segmesh::Float3> points)
{
    std::vector<double> parameters(points.size(), 0.0);
    for (std::size_t index = 1; index < points.size(); ++index)
    {
        parameters[index] = parameters[index - 1] + pointDistance(points[index - 1], points[index]);
    }

    const double totalLength = parameters.back();
    if (totalLength > kCoincidentPointEpsilon)
    {
        for (double& parameter : parameters)
        {
            parameter /= totalLength;
        }
    }
    return parameters;
}

std::array<segmesh::Float3, 4> lineEquivalentCubic(
    const segmesh::Float3& first,
    const segmesh::Float3& last
)
{
    const Eigen::Vector3d start = toEigen(first);
    const Eigen::Vector3d end = toEigen(last);
    const Eigen::Vector3d difference = end - start;
    return {
        first,
        toFloat3(start + difference / 3.0),
        toFloat3(start + difference * (2.0 / 3.0)),
        last,
    };
}

double pointToLineDistance(
    const segmesh::Float3& point,
    const segmesh::Float3& lineStart,
    const segmesh::Float3& lineEnd
)
{
    const Eigen::Vector3d start = toEigen(lineStart);
    const Eigen::Vector3d direction = toEigen(lineEnd) - start;
    const double squaredLength = direction.squaredNorm();
    if (squaredLength <= kCoincidentPointEpsilon * kCoincidentPointEpsilon)
    {
        return (toEigen(point) - start).norm();
    }
    return ((toEigen(point) - start).cross(direction)).norm() / std::sqrt(squaredLength);
}

double pointToSegmentDistance(
    const segmesh::Float3& point,
    const segmesh::Float3& segmentStart,
    const segmesh::Float3& segmentEnd
)
{
    const Eigen::Vector3d start = toEigen(segmentStart);
    const Eigen::Vector3d direction = toEigen(segmentEnd) - start;
    const double squaredLength = direction.squaredNorm();
    if (squaredLength <= kCoincidentPointEpsilon * kCoincidentPointEpsilon)
    {
        return (toEigen(point) - start).norm();
    }
    const double parameter = std::clamp(
        (toEigen(point) - start).dot(direction) / squaredLength,
        0.0,
        1.0
    );
    return (toEigen(point) - (start + parameter * direction)).norm();
}

void sampleCubicForFit(
    const std::array<segmesh::Float3, 4>& cubic,
    double samplingTolerance,
    uint32_t depth,
    std::vector<segmesh::Float3>& outPoints
)
{
    if ((depth >= kMinimumFitSamplingDepth && detail::cubicFlatness(cubic) <= samplingTolerance)
        || depth == kMaximumFitSamplingDepth)
    {
        outPoints.push_back(cubic.back());
        return;
    }

    std::array<segmesh::Float3, 4> left{};
    std::array<segmesh::Float3, 4> right{};
    detail::splitCubicAtHalf(cubic, left, right);
    sampleCubicForFit(left, samplingTolerance, depth + 1, outPoints);
    sampleCubicForFit(right, samplingTolerance, depth + 1, outPoints);
}

double curveToPolylineDeviation(
    const std::array<segmesh::Float3, 4>& cubic,
    std::span<const segmesh::Float3> polyline,
    double fitTolerance,
    std::size_t& outSplitPoint
)
{
    std::vector<segmesh::Float3> curveSamples;
    curveSamples.reserve(65);
    curveSamples.push_back(cubic.front());
    const double chordLength = pointDistance(cubic.front(), cubic.back());
    const double samplingTolerance = std::max(
        std::min(fitTolerance * 0.125, chordLength * 1.0e-3),
        kCoincidentPointEpsilon
    );
    sampleCubicForFit(cubic, samplingTolerance, 0, curveSamples);

    double maximumDistance = 0.0;
    std::size_t nearestSegmentAtMaximum = 0;
    for (const segmesh::Float3& curvePoint : curveSamples)
    {
        double nearestDistance = std::numeric_limits<double>::infinity();
        std::size_t nearestSegment = 0;
        for (std::size_t segment = 0; segment + 1 < polyline.size(); ++segment)
        {
            const double distance = pointToSegmentDistance(
                curvePoint,
                polyline[segment],
                polyline[segment + 1]
            );
            if (distance < nearestDistance)
            {
                nearestDistance = distance;
                nearestSegment = segment;
            }
        }
        if (nearestDistance > maximumDistance)
        {
            maximumDistance = nearestDistance;
            nearestSegmentAtMaximum = nearestSegment;
        }
    }

    if (polyline.size() > 2)
    {
        outSplitPoint = std::clamp(
            nearestSegmentAtMaximum + 1,
            std::size_t{1},
            polyline.size() - 2
        );
    }
    else
    {
        outSplitPoint = 0;
    }
    return maximumDistance;
}

std::array<segmesh::Float3, 4> tangentCubicForSegment(
    const segmesh::Float3& first,
    const segmesh::Float3& last,
    Eigen::Vector3d startTangent,
    Eigen::Vector3d endTangent,
    double handleLength
)
{
    const Eigen::Vector3d start = toEigen(first);
    const Eigen::Vector3d end = toEigen(last);
    const Eigen::Vector3d chord = end - start;
    const Eigen::Vector3d chordDirection = chord.normalized();
    if (!startTangent.allFinite() || startTangent.dot(chordDirection) <= 0.0)
    {
        startTangent = chordDirection;
    }
    if (!endTangent.allFinite() || endTangent.dot(chordDirection) <= 0.0)
    {
        endTangent = chordDirection;
    }
    return {
        first,
        toFloat3(start + handleLength * startTangent.normalized()),
        toFloat3(end - handleLength * endTangent.normalized()),
        last,
    };
}

bool fitCubicWithTangents(
    std::span<const segmesh::Float3> points,
    const Eigen::Vector3d& startTangent,
    const Eigen::Vector3d& endTangent,
    std::array<segmesh::Float3, 4>& outControlPoints,
    segmesh::BorderCurveFitStats& outStats,
    std::size_t& outWorstPoint,
    std::string& error
)
{
    outControlPoints = {};
    outStats = {};
    outWorstPoint = 0;

    if (points.size() < 2)
    {
        error = "Bezier fitting requires at least two sample points.";
        return false;
    }
    if (!std::all_of(points.begin(), points.end(), isFinite)
        || !startTangent.allFinite() || !endTangent.allFinite())
    {
        error = "Bezier fitting received non-finite input.";
        return false;
    }
    if (points.size() == 2)
    {
        const double chordLength = pointDistance(points.front(), points.back());
        outControlPoints = tangentCubicForSegment(
            points.front(), points.back(), startTangent, endTangent, chordLength / 3.0
        );
        return true;
    }

    const std::vector<double> parameters = chordLengthParameters(points);
    const Eigen::Vector3d first = toEigen(points.front());
    const Eigen::Vector3d last = toEigen(points.back());
    const double chordLength = pointDistance(points.front(), points.back());

    const Eigen::Index rowCount = static_cast<Eigen::Index>((points.size() - 2) * 3);
    Eigen::MatrixXd design(rowCount, 2);
    Eigen::VectorXd rightHandSide(rowCount);
    for (std::size_t index = 1; index + 1 < points.size(); ++index)
    {
        const double t = parameters[index];
        const double oneMinusT = 1.0 - t;
        const double b0 = oneMinusT * oneMinusT * oneMinusT;
        const double b1 = 3.0 * oneMinusT * oneMinusT * t;
        const double b2 = 3.0 * oneMinusT * t * t;
        const double b3 = t * t * t;
        const Eigen::Index row = static_cast<Eigen::Index>((index - 1) * 3);
        design.block<3, 1>(row, 0) = b1 * startTangent;
        design.block<3, 1>(row, 1) = -b2 * endTangent;
        rightHandSide.segment<3>(row) = toEigen(points[index])
            - (b0 + b1) * first - (b2 + b3) * last;
    }

    const double fallbackHandle = chordLength / 3.0;
    double startHandle = fallbackHandle;
    double endHandle = fallbackHandle;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(design);
    const Eigen::VectorXd solution = solver.solve(rightHandSide);
    if (solver.rank() == 2 && solution.allFinite())
    {
        const double minimumHandle = std::max(chordLength * 1.0e-6, kCoincidentPointEpsilon);
        const double maximumHandle = std::max(chordLength, minimumHandle);
        if (solution[0] >= minimumHandle && solution[0] <= maximumHandle
            && solution[1] >= minimumHandle && solution[1] <= maximumHandle)
        {
            startHandle = solution[0];
            endHandle = solution[1];
        }
    }

    outControlPoints = {
        points.front(),
        toFloat3(first + startHandle * startTangent),
        toFloat3(last - endHandle * endTangent),
        points.back(),
    };
    if (!std::all_of(outControlPoints.begin(), outControlPoints.end(), isFinite))
    {
        outControlPoints = {};
        error = "Bezier fitting produced non-finite control points.";
        return false;
    }

    const std::span<const segmesh::Float3> cubic(outControlPoints);
    double squaredErrorSum = 0.0;
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const double fitError = pointDistance(
            points[index],
            segmesh::evaluateBezier(cubic, parameters[index])
        );
        if (!std::isfinite(fitError))
        {
            outControlPoints = {};
            outStats = {};
            error = "Bezier fitting produced a non-finite residual.";
            return false;
        }
        if (fitError > outStats.maxError)
        {
            outStats.maxError = fitError;
            outWorstPoint = index;
        }
        squaredErrorSum += fitError * fitError;
    }
    outStats.rmsError = std::sqrt(squaredErrorSum / static_cast<double>(points.size()));
    return true;
}
}

namespace detail
{
std::vector<segmesh::Float3> removeConsecutiveCoincidentPoints(
    std::span<const segmesh::Float3> points
)
{
    std::vector<segmesh::Float3> result;
    result.reserve(points.size());
    for (const segmesh::Float3& point : points)
    {
        if (result.empty() || pointDistance(result.back(), point) > kCoincidentPointEpsilon)
        {
            result.push_back(point);
        }
    }
    return result;
}

Eigen::Vector3d forwardTangentAt(
    std::span<const segmesh::Float3> points,
    std::size_t index
)
{
    Eigen::Vector3d tangent;
    if (index == 0)
    {
        tangent = toEigen(points[1]) - toEigen(points[0]);
    }
    else if (index + 1 == points.size())
    {
        tangent = toEigen(points[index]) - toEigen(points[index - 1]);
    }
    else
    {
        const Eigen::Vector3d incoming = (
            toEigen(points[index]) - toEigen(points[index - 1])
        ).normalized();
        const Eigen::Vector3d outgoing = (
            toEigen(points[index + 1]) - toEigen(points[index])
        ).normalized();
        tangent = incoming + outgoing;
        if (tangent.squaredNorm() <= kCoincidentPointEpsilon * kCoincidentPointEpsilon)
        {
            tangent = outgoing;
        }
    }
    return tangent.normalized();
}

double cubicFlatness(const std::array<segmesh::Float3, 4>& cubic)
{
    const double perpendicularDeviation = std::max(
        pointToLineDistance(cubic[1], cubic[0], cubic[3]),
        pointToLineDistance(cubic[2], cubic[0], cubic[3])
    );
    const double controlPolygonLength = pointDistance(cubic[0], cubic[1])
        + pointDistance(cubic[1], cubic[2])
        + pointDistance(cubic[2], cubic[3]);
    const double chordLength = pointDistance(cubic[0], cubic[3]);
    return std::max(perpendicularDeviation, controlPolygonLength - chordLength);
}

void splitCubicAtHalf(
    const std::array<segmesh::Float3, 4>& cubic,
    std::array<segmesh::Float3, 4>& left,
    std::array<segmesh::Float3, 4>& right
)
{
    const Eigen::Vector3d p0 = toEigen(cubic[0]);
    const Eigen::Vector3d p1 = toEigen(cubic[1]);
    const Eigen::Vector3d p2 = toEigen(cubic[2]);
    const Eigen::Vector3d p3 = toEigen(cubic[3]);
    const Eigen::Vector3d p01 = (p0 + p1) * 0.5;
    const Eigen::Vector3d p12 = (p1 + p2) * 0.5;
    const Eigen::Vector3d p23 = (p2 + p3) * 0.5;
    const Eigen::Vector3d p012 = (p01 + p12) * 0.5;
    const Eigen::Vector3d p123 = (p12 + p23) * 0.5;
    const Eigen::Vector3d midpoint = (p012 + p123) * 0.5;
    left = {cubic[0], toFloat3(p01), toFloat3(p012), toFloat3(midpoint)};
    right = {toFloat3(midpoint), toFloat3(p123), toFloat3(p23), cubic[3]};
}

bool fitAdaptiveCubicPieces(
    std::span<const segmesh::Float3> points,
    double fitTolerance,
    const Eigen::Vector3d& chainStartTangent,
    const Eigen::Vector3d& chainEndTangent,
    std::vector<CubicPiece>& outPieces,
    std::string& error
)
{
    struct PendingPiece
    {
        std::size_t first = 0;
        std::size_t last = 0;
        Eigen::Vector3d startTangent{};
        Eigen::Vector3d endTangent{};
    };

    std::vector<PendingPiece> pending;
    pending.push_back({
        0,
        points.size() - 1,
        chainStartTangent,
        chainEndTangent,
    });

    while (!pending.empty())
    {
        PendingPiece current = pending.back();
        pending.pop_back();
        const std::span<const segmesh::Float3> piecePoints = points.subspan(
            current.first,
            current.last - current.first + 1
        );

        CubicPiece fitted;
        fitted.firstSample = current.first;
        fitted.lastSample = current.last;
        std::size_t worstLocalPoint = 0;
        if (!fitCubicWithTangents(
                piecePoints,
                current.startTangent,
                current.endTangent,
                fitted.controlPoints,
                fitted.fitStats,
                worstLocalPoint,
                error))
        {
            return false;
        }

        std::size_t geometricSplitPoint = 0;
        double geometricDeviation = curveToPolylineDeviation(
            fitted.controlPoints,
            piecePoints,
            fitTolerance,
            geometricSplitPoint
        );

        if (piecePoints.size() == 2)
        {
            const double chordLength = pointDistance(piecePoints.front(), piecePoints.back());
            double handleLength = chordLength / 3.0;
            for (uint32_t attempt = 0;
                 geometricDeviation > fitTolerance && attempt < 24;
                 ++attempt)
            {
                handleLength *= 0.5;
                fitted.controlPoints = tangentCubicForSegment(
                    piecePoints.front(),
                    piecePoints.back(),
                    current.startTangent,
                    current.endTangent,
                    handleLength
                );
                geometricDeviation = curveToPolylineDeviation(
                    fitted.controlPoints,
                    piecePoints,
                    fitTolerance,
                    geometricSplitPoint
                );
            }
            if (geometricDeviation > fitTolerance)
            {
                fitted.controlPoints = lineEquivalentCubic(
                    piecePoints.front(), piecePoints.back()
                );
                geometricDeviation = 0.0;
            }
            fitted.fitStats.maxError = std::max(
                fitted.fitStats.maxError,
                geometricDeviation
            );
            outPieces.push_back(std::move(fitted));
            continue;
        }

        if (geometricDeviation > fitted.fitStats.maxError)
        {
            fitted.fitStats.maxError = geometricDeviation;
            worstLocalPoint = geometricSplitPoint;
        }

        if (fitted.fitStats.maxError <= fitTolerance)
        {
            outPieces.push_back(std::move(fitted));
            continue;
        }

        if (worstLocalPoint == 0 || worstLocalPoint + 1 == piecePoints.size())
        {
            worstLocalPoint = piecePoints.size() / 2;
        }
        const std::size_t split = current.first + worstLocalPoint;
        const Eigen::Vector3d splitTangent = forwardTangentAt(points, split);

        // Push right first so the LIFO traversal emits pieces in chain order.
        pending.push_back({split, current.last, splitTangent, current.endTangent});
        pending.push_back({current.first, split, current.startTangent, splitTangent});
    }
    return true;
}
}

Float3 evaluateBezier(std::span<const Float3> controlPoints, double t)
{
    if (controlPoints.empty())
    {
        return {};
    }
    if (controlPoints.size() == 1 || t <= 0.0)
    {
        return controlPoints.front();
    }
    if (t >= 1.0)
    {
        return controlPoints.back();
    }

    std::vector<Eigen::Vector3d> work;
    work.reserve(controlPoints.size());
    for (const Float3& point : controlPoints)
    {
        work.push_back(toEigen(point));
    }

    for (std::size_t remaining = work.size() - 1; remaining > 0; --remaining)
    {
        for (std::size_t index = 0; index < remaining; ++index)
        {
            work[index] = (1.0 - t) * work[index] + t * work[index + 1];
        }
    }
    return toFloat3(work.front());
}

bool fitEndpointConstrainedCubic(
    std::span<const Float3> points,
    std::array<Float3, 4>& outControlPoints,
    BorderCurveFitStats& outStats,
    std::string& error
)
{
    error.clear();
    outControlPoints = {};
    outStats = {};

    for (const Float3& point : points)
    {
        if (!isFinite(point))
        {
            error = "Bezier fitting received a non-finite sample point.";
            return false;
        }
    }

    const std::vector<Float3> samples = detail::removeConsecutiveCoincidentPoints(points);
    if (samples.size() < 2)
    {
        error = "Bezier fitting requires at least two distinct consecutive sample points.";
        return false;
    }

    std::size_t worstPoint = 0;
    return fitCubicWithTangents(
        samples,
        detail::forwardTangentAt(samples, 0),
        detail::forwardTangentAt(samples, samples.size() - 1),
        outControlPoints,
        outStats,
        worstPoint,
        error
    );
}
}
