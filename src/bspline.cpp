#include "bspline.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace
{
constexpr std::size_t kDegree = segmesh::CubicBSpline::degree;
constexpr double kPointEpsilon = 1.0e-12;
constexpr uint32_t kMaximumControlPointCount = 128;
constexpr uint32_t kMaximumErrorBoundDepth = 8;

using Cubic = std::array<Eigen::Vector3d, 4>;

Eigen::Vector3d toEigen(const segmesh::Float3& point)
{
    return {point.x, point.y, point.z};
}

segmesh::Float3 toFloat3(const Eigen::Vector3d& point)
{
    return {static_cast<float>(point.x()), static_cast<float>(point.y()), static_cast<float>(point.z())};
}

bool isFinite(const segmesh::Float3& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

bool validateCurve(const segmesh::CubicBSpline& curve, std::string& error)
{
    const std::size_t count = curve.controlPoints.size();
    const std::size_t extraKnots = curve.periodic ? 7 : 4;
    if (count < 4 || count > std::numeric_limits<std::size_t>::max() - extraKnots
        || curve.knots.size() != count + extraKnots)
    {
        error = "Cubic B-spline control and knot counts are inconsistent.";
        return false;
    }
    if (!std::all_of(curve.controlPoints.begin(), curve.controlPoints.end(), isFinite)
        || !std::all_of(curve.knots.begin(), curve.knots.end(), [](double knot) { return std::isfinite(knot); })
        || !std::is_sorted(curve.knots.begin(), curve.knots.end()))
    {
        error = "B-spline controls must be finite and knots must be finite and ordered.";
        return false;
    }

    const std::size_t effectiveCount = count + (curve.periodic ? kDegree : 0);
    if (curve.knots[kDegree] != 0.0 || curve.knots[effectiveCount] != 1.0)
    {
        error = "B-spline parameter domain must be normalized to zero through one.";
        return false;
    }
    for (std::size_t span = kDegree; span < effectiveCount; ++span)
    {
        if (curve.knots[span + 1] <= curve.knots[span])
        {
            error = "B-spline interior knots must be simple.";
            return false;
        }
    }
    for (std::size_t index = 0; index < kDegree; ++index)
    {
        if (curve.periodic)
        {
            if (curve.knots[index + 1] <= curve.knots[index]
                || curve.knots[count + 4 + index] <= curve.knots[count + 3 + index]
                || std::abs(curve.knots[index] - (curve.knots[count + index] - 1.0)) > kPointEpsilon
                || std::abs(curve.knots[count + 4 + index] - (curve.knots[4 + index] + 1.0)) > kPointEpsilon)
            {
                error = "Periodic B-spline knots must continue the intervals across the seam.";
                return false;
            }
        }
        else if (curve.knots[index] != 0.0 || curve.knots[count + 1 + index] != 1.0)
        {
            error = "Open cubic B-splines require four repeated knots at each endpoint.";
            return false;
        }
    }
    return true;
}

struct LocalBasis
{
    std::array<std::size_t, 4> controls{};
    std::array<double, 4> values{};
};

LocalBasis basisAt(const segmesh::CubicBSpline& curve, double parameter)
{
    const double t = curve.periodic ? parameter - std::floor(parameter) : std::clamp(parameter, 0.0, 1.0);
    const std::size_t count = curve.controlPoints.size();
    const std::size_t effectiveCount = count + (curve.periodic ? kDegree : 0);
    const auto upper = std::upper_bound(curve.knots.begin() + kDegree, curve.knots.begin() + effectiveCount + 1, t);
    const std::size_t span = std::clamp(
        static_cast<std::size_t>(upper - curve.knots.begin() - 1), kDegree, effectiveCount - 1
    );

    LocalBasis basis;
    std::array<double, 4> left{};
    std::array<double, 4> right{};
    basis.values[0] = 1.0;
    for (std::size_t degree = 1; degree <= kDegree; ++degree)
    {
        left[degree] = t - curve.knots[span + 1 - degree];
        right[degree] = curve.knots[span + degree] - t;
        double saved = 0.0;
        for (std::size_t index = 0; index < degree; ++index)
        {
            const double denominator = right[index + 1] + left[degree - index];
            const double term = denominator > 0.0 ? basis.values[index] / denominator : 0.0;
            basis.values[index] = saved + right[index + 1] * term;
            saved = left[degree - index] * term;
        }
        basis.values[degree] = saved;
    }
    for (std::size_t index = 0; index <= kDegree; ++index)
    {
        basis.controls[index] = (span - kDegree + index) % count;
    }
    return basis;
}

Eigen::Vector3d evaluate(const segmesh::CubicBSpline& curve, double parameter)
{
    const LocalBasis basis = basisAt(curve, parameter);
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    for (std::size_t index = 0; index <= kDegree; ++index)
    {
        point += basis.values[index] * toEigen(curve.controlPoints[basis.controls[index]]);
    }
    return point;
}

std::vector<double> makeKnots(const std::vector<double>& breaks, bool periodic)
{
    std::vector<double> knots;
    knots.reserve(breaks.size() + 6);
    if (periodic)
    {
        const std::size_t spanCount = breaks.size() - 1;
        for (std::size_t index = spanCount - kDegree; index < spanCount; ++index)
        {
            knots.push_back(breaks[index] - 1.0);
        }
        knots.insert(knots.end(), breaks.begin(), breaks.end());
        for (std::size_t index = 1; index <= kDegree; ++index)
        {
            knots.push_back(breaks[index] + 1.0);
        }
    }
    else
    {
        knots.insert(knots.end(), 3, 0.0);
        knots.insert(knots.end(), breaks.begin(), breaks.end());
        knots.insert(knots.end(), 3, 1.0);
    }
    return knots;
}

Cubic bezierControls(const segmesh::CubicBSpline& curve, double first, double last)
{
    const Eigen::Vector3d start = evaluate(curve, first);
    const Eigen::Vector3d end = evaluate(curve, last);
    const double length = last - first;
    const Eigen::Vector3d value1 = evaluate(curve, first + length / 3.0) - (8.0 * start + end) / 27.0;
    const Eigen::Vector3d value2 = evaluate(curve, first + 2.0 * length / 3.0) - (start + 8.0 * end) / 27.0;
    return {start, 3.0 * value1 - 1.5 * value2, 3.0 * value2 - 1.5 * value1, end};
}

struct Guide
{
    std::vector<segmesh::Float3> points;
    std::vector<double> parameters;
    double length = 0.0;
    bool periodic = false;
};

bool prepareGuide(std::span<const segmesh::Float3> points, bool periodic, Guide& guide, std::string& error)
{
    guide.periodic = periodic;
    for (const segmesh::Float3& point : points)
    {
        if (!isFinite(point))
        {
            error = "B-spline fitting received a non-finite guide point.";
            return false;
        }
        if (guide.points.empty() || (toEigen(point) - toEigen(guide.points.back())).norm() > kPointEpsilon)
        {
            guide.points.push_back(point);
        }
    }
    if (periodic && guide.points.size() > 1
        && (toEigen(guide.points.back()) - toEigen(guide.points.front())).norm() <= kPointEpsilon)
    {
        guide.points.pop_back();
    }
    if (guide.points.size() < (periodic ? 4u : 2u)
        || guide.points.size() > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()))
    {
        error = "B-spline fitting requires two open guide points or four periodic guide points within the supported size.";
        return false;
    }

    guide.parameters.resize(guide.points.size(), 0.0);
    for (std::size_t index = 1; index < guide.points.size(); ++index)
    {
        guide.length += (toEigen(guide.points[index]) - toEigen(guide.points[index - 1])).norm();
        guide.parameters[index] = guide.length;
    }
    if (periodic)
    {
        guide.length += (toEigen(guide.points.front()) - toEigen(guide.points.back())).norm();
    }
    if (!std::isfinite(guide.length) || guide.length <= kPointEpsilon)
    {
        error = "B-spline guide length must be finite and positive.";
        return false;
    }
    for (std::size_t index = 0; index < guide.parameters.size(); ++index)
    {
        guide.parameters[index] /= guide.length;
        if ((index > 0 && guide.parameters[index] <= guide.parameters[index - 1])
            || (periodic && guide.parameters[index] >= 1.0))
        {
            error = "B-spline guide spacing exceeds arc-length parameter precision.";
            return false;
        }
    }
    return true;
}

Eigen::Vector3d guideAt(const Guide& guide, double parameter)
{
    const auto upper = std::upper_bound(guide.parameters.begin(), guide.parameters.end(), parameter);
    if (upper == guide.parameters.begin())
    {
        return toEigen(guide.points.front());
    }
    const std::size_t first = static_cast<std::size_t>(upper - guide.parameters.begin() - 1);
    const std::size_t next = first + 1;
    if (next == guide.points.size() && !guide.periodic)
    {
        return toEigen(guide.points.back());
    }
    const double lastParameter = next == guide.points.size() ? 1.0 : guide.parameters[next];
    const double fraction = std::clamp(
        (parameter - guide.parameters[first]) / (lastParameter - guide.parameters[first]), 0.0, 1.0
    );
    return (1.0 - fraction) * toEigen(guide.points[first]) + fraction * toEigen(guide.points[next % guide.points.size()]);
}

bool fitWithKnots(const Guide& guide, const std::vector<double>& breaks, segmesh::CubicBSpline& curve, std::string& error)
{
    curve.periodic = guide.periodic;
    curve.knots = makeKnots(breaks, guide.periodic);
    const std::size_t count = guide.periodic ? breaks.size() - 1 : breaks.size() + 2;
    curve.controlPoints.resize(count);
    if (!guide.periodic)
    {
        curve.controlPoints.front() = guide.points.front();
        curve.controlPoints.back() = guide.points.back();
        if (guide.points.size() <= 3)
        {
            const Eigen::Vector3d start = toEigen(guide.points.front());
            const Eigen::Vector3d end = toEigen(guide.points.back());
            if (guide.points.size() == 2)
            {
                curve.controlPoints[1] = toFloat3((2.0 * start + end) / 3.0);
                curve.controlPoints[2] = toFloat3((start + 2.0 * end) / 3.0);
            }
            else
            {
                const double t = guide.parameters[1];
                const Eigen::Vector3d middle = (toEigen(guide.points[1]) - (1.0 - t) * (1.0 - t) * start - t * t * end)
                    / (2.0 * t * (1.0 - t));
                curve.controlPoints[1] = toFloat3((start + 2.0 * middle) / 3.0);
                curve.controlPoints[2] = toFloat3((end + 2.0 * middle) / 3.0);
            }
            return validateCurve(curve, error);
        }
    }

    const std::size_t fixedCount = guide.periodic ? 0 : 2;
    const Eigen::Index rowCount = static_cast<Eigen::Index>(guide.points.size() - fixedCount);
    const Eigen::Index unknownCount = static_cast<Eigen::Index>(count - fixedCount);
    if (rowCount < unknownCount)
    {
        error = "B-spline knot refinement has more controls than supported guide samples.";
        return false;
    }
    Eigen::MatrixXd design = Eigen::MatrixXd::Zero(rowCount, unknownCount);
    Eigen::MatrixXd rhs(rowCount, 3);
    for (Eigen::Index row = 0; row < rowCount; ++row)
    {
        const std::size_t sample = static_cast<std::size_t>(row) + (guide.periodic ? 0 : 1);
        Eigen::Vector3d value = toEigen(guide.points[sample]);
        const LocalBasis basis = basisAt(curve, guide.parameters[sample]);
        for (std::size_t index = 0; index <= kDegree; ++index)
        {
            const std::size_t control = basis.controls[index];
            const double coefficient = basis.values[index];
            if (!guide.periodic && (control == 0 || control + 1 == count))
            {
                value -= coefficient * toEigen(curve.controlPoints[control]);
            }
            else
            {
                const Eigen::Index column = static_cast<Eigen::Index>(control - (guide.periodic ? 0 : 1));
                design(row, column) += coefficient;
            }
        }
        rhs.row(row) = value.transpose();
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(design);
    solver.setThreshold(1.0e-12);
    if (solver.rank() != unknownCount)
    {
        error = "B-spline knots are not sufficiently supported by the guide samples.";
        return false;
    }
    const Eigen::MatrixXd solution = solver.solve(rhs);
    if (!solution.allFinite())
    {
        error = "B-spline least-squares fitting produced non-finite controls.";
        return false;
    }
    for (Eigen::Index row = 0; row < unknownCount; ++row)
    {
        const std::size_t control = static_cast<std::size_t>(row) + (guide.periodic ? 0 : 1);
        curve.controlPoints[control] = toFloat3(solution.row(row).transpose());
    }
    return validateCurve(curve, error);
}

double residualBound(const Cubic& controls, double tolerance, uint32_t depth = 0)
{
    double upper = 0.0;
    for (const Eigen::Vector3d& point : controls)
    {
        upper = std::max(upper, point.norm());
    }
    const Eigen::Vector3d p01 = 0.5 * (controls[0] + controls[1]);
    const Eigen::Vector3d p12 = 0.5 * (controls[1] + controls[2]);
    const Eigen::Vector3d p23 = 0.5 * (controls[2] + controls[3]);
    const Eigen::Vector3d p012 = 0.5 * (p01 + p12);
    const Eigen::Vector3d p123 = 0.5 * (p12 + p23);
    const Eigen::Vector3d middle = 0.5 * (p012 + p123);
    const double lower = std::max({controls.front().norm(), controls.back().norm(), middle.norm()});
    if (upper <= tolerance || upper - lower <= std::max(tolerance * 0.1, upper * 0.01)
        || depth == kMaximumErrorBoundDepth)
    {
        return upper;
    }
    return std::max(
        residualBound({controls[0], p01, p012, middle}, tolerance, depth + 1),
        residualBound({middle, p123, p23, controls[3]}, tolerance, depth + 1)
    );
}

std::size_t intervalAt(const std::vector<double>& breaks, double parameter)
{
    const auto upper = std::upper_bound(breaks.begin(), breaks.end(), parameter);
    return std::min(static_cast<std::size_t>(upper - breaks.begin() - 1), breaks.size() - 2);
}

segmesh::BSplineFitStats measureFit(
    const Guide& guide,
    const segmesh::CubicBSpline& curve,
    const std::vector<double>& breaks,
    double tolerance,
    std::vector<double>& spanErrors
)
{
    segmesh::BSplineFitStats stats;
    double squaredError = 0.0;
    spanErrors.assign(breaks.size() - 1, 0.0);
    for (std::size_t sample = 0; sample < guide.points.size(); ++sample)
    {
        const double deviation = (evaluate(curve, guide.parameters[sample]) - toEigen(guide.points[sample])).norm();
        stats.maxSampleError = std::max(stats.maxSampleError, deviation);
        squaredError += deviation * deviation;
        const std::size_t span = intervalAt(breaks, guide.parameters[sample]);
        spanErrors[span] = std::max(spanErrors[span], deviation);
    }
    stats.rmsError = std::sqrt(squaredError / static_cast<double>(guide.points.size()));

    std::vector<double> boundaries = guide.parameters;
    boundaries.insert(boundaries.end(), breaks.begin(), breaks.end());
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    for (std::size_t interval = 1; interval < boundaries.size(); ++interval)
    {
        const double first = boundaries[interval - 1];
        const double last = boundaries[interval];
        Cubic residual = bezierControls(curve, first, last);
        const Eigen::Vector3d guideStart = guideAt(guide, first);
        const Eigen::Vector3d guideEnd = guideAt(guide, last);
        for (std::size_t control = 0; control <= kDegree; ++control)
        {
            const double fraction = static_cast<double>(control) / 3.0;
            residual[control] -= (1.0 - fraction) * guideStart + fraction * guideEnd;
        }
        const double bound = residualBound(residual, tolerance);
        const std::size_t span = intervalAt(breaks, first + 0.5 * (last - first));
        spanErrors[span] = std::max(spanErrors[span], bound);
    }
    stats.maxError = *std::max_element(spanErrors.begin(), spanErrors.end());
    return stats;
}
}

namespace segmesh
{
bool evaluateBSpline(const CubicBSpline& curve, double parameter, Float3& outPoint, std::string& error)
{
    error.clear();
    if (!std::isfinite(parameter))
    {
        error = "B-spline evaluation requires a finite parameter.";
        return false;
    }
    if (!validateCurve(curve, error))
    {
        return false;
    }
    const Float3 point = toFloat3(evaluate(curve, parameter));
    if (!isFinite(point))
    {
        error = "B-spline evaluation produced a non-finite point.";
        return false;
    }
    outPoint = point;
    return true;
}

bool convertBSplineToBezierSpans(const CubicBSpline& curve, std::vector<BSplineBezierSpan>& outSpans, std::string& error)
{
    error.clear();
    if (!validateCurve(curve, error))
    {
        return false;
    }
    const std::size_t effectiveCount = curve.controlPoints.size() + (curve.periodic ? kDegree : 0);
    std::vector<BSplineBezierSpan> spans;
    spans.reserve(effectiveCount - kDegree);
    for (std::size_t span = kDegree; span < effectiveCount; ++span)
    {
        BSplineBezierSpan piece;
        piece.firstParameter = curve.knots[span];
        piece.lastParameter = curve.knots[span + 1];
        const Cubic controls = bezierControls(curve, piece.firstParameter, piece.lastParameter);
        for (std::size_t index = 0; index <= kDegree; ++index)
        {
            piece.controlPoints[index] = toFloat3(controls[index]);
            if (!isFinite(piece.controlPoints[index]))
            {
                error = "B-spline conversion produced non-finite Bezier controls.";
                return false;
            }
        }
        spans.push_back(piece);
    }
    outSpans = std::move(spans);
    return true;
}

bool fitAdaptiveCubicBSpline(
    std::span<const Float3> guidePoints,
    bool periodic,
    const BSplineFitSettings& settings,
    CubicBSpline& outCurve,
    BSplineFitStats& outStats,
    std::string& error
)
{
    error.clear();
    if (!std::isfinite(settings.fitTolerance) || settings.fitTolerance < 0.0
        || !std::isfinite(settings.minimumSpanLength) || settings.minimumSpanLength < 0.0
        || settings.maxControlPointCount < 4 || settings.maxControlPointCount > kMaximumControlPointCount)
    {
        error = "B-spline fitting needs finite non-negative tolerances and a control limit between 4 and 128.";
        return false;
    }
    Guide guide;
    if (!prepareGuide(guidePoints, periodic, guide, error))
    {
        return false;
    }
    std::vector<double> breaks = periodic
        ? std::vector<double>{0.0, 0.25, 0.5, 0.75, 1.0}
        : std::vector<double>{0.0, 1.0};
    CubicBSpline current;
    if (!fitWithKnots(guide, breaks, current, error))
    {
        return false;
    }
    std::vector<double> spanErrors;
    BSplineFitStats currentStats = measureFit(guide, current, breaks, settings.fitTolerance, spanErrors);
    CubicBSpline bestCurve = current;
    BSplineFitStats bestStats = currentStats;
    uint32_t refinements = 0;
    const double minimumParameterSpan = std::max(
        settings.minimumSpanLength / guide.length, 64.0 * std::numeric_limits<double>::epsilon()
    );

    while (bestStats.maxError > settings.fitTolerance
        && current.controlPoints.size() < settings.maxControlPointCount
        && current.controlPoints.size() < guide.points.size())
    {
        std::vector<std::size_t> candidates(spanErrors.size());
        std::iota(candidates.begin(), candidates.end(), 0);
        std::stable_sort(candidates.begin(), candidates.end(), [&](std::size_t first, std::size_t second)
        {
            return spanErrors[first] > spanErrors[second];
        });
        bool refined = false;
        for (const std::size_t span : candidates)
        {
            if (spanErrors[span] <= settings.fitTolerance)
            {
                break;
            }
            const double midpoint = breaks[span] + 0.5 * (breaks[span + 1] - breaks[span]);
            if (midpoint - breaks[span] < minimumParameterSpan
                || breaks[span + 1] - midpoint < minimumParameterSpan)
            {
                continue;
            }
            std::vector<double> refinedBreaks = breaks;
            refinedBreaks.insert(refinedBreaks.begin() + static_cast<std::ptrdiff_t>(span + 1), midpoint);
            CubicBSpline candidate;
            std::string refinementError;
            if (!fitWithKnots(guide, refinedBreaks, candidate, refinementError))
            {
                continue;
            }
            breaks = std::move(refinedBreaks);
            current = std::move(candidate);
            ++refinements;
            currentStats = measureFit(guide, current, breaks, settings.fitTolerance, spanErrors);
            currentStats.refinementCount = refinements;
            if (currentStats.maxError < bestStats.maxError
                || (currentStats.maxError == bestStats.maxError && currentStats.rmsError < bestStats.rmsError))
            {
                bestCurve = current;
                bestStats = currentStats;
            }
            refined = true;
            break;
        }
        if (!refined)
        {
            break;
        }
    }

    bestStats.refinementCapped = bestStats.maxError > settings.fitTolerance;
    outCurve = std::move(bestCurve);
    outStats = bestStats;
    return true;
}
}
