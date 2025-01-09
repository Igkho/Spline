#include "SplineSolver.h"
#include "SplineD.h"
#include "Distance.h"
#include <algorithm>
#include <stdexcept>
#include "Eigen/Eigen"

namespace spline {

//! A method for calculation of all spline data
template <class T, int Degree>
std::tuple<T, std::vector<T>, std::vector<T>, std::vector<T>>
SplineSolver<T, Degree>::SolveSpline(const std::vector<std::vector<T>> &ref_points) {
    if (ref_points.size() <= Degree) {
        throw std::runtime_error("Points count should be higher than spline degree");
    }
    // Check points validity
    for (const auto &p : ref_points) {
        if (p.size() != Spline<T, Degree>::Dim) {
            throw std::runtime_error("All points sizes should be equal to " +
                                     std::to_string(Spline<T, Degree>::Dim));
        }
    }
    // Calculate chord length parameters
    auto [min_dparam, t] = CalculateParams(ref_points);
    // Calculate knots
    auto knots = CalculateKnots(t);
    // Calculate coefficients
    std::vector<T> N = CalculateCoeffsMatrix(t, knots);
    // Solve the spline
    auto [P, Q] = Solve(N, knots, ref_points);
    return {min_dparam, knots, P, Q};
}

//! A method for calculation of t parameters and minimum parameters distance
template <class T, int Degree>
std::tuple<T, std::vector<T>>
SplineSolver<T, Degree>::CalculateParams(const std::vector<std::vector<T>> &ref_points) {
    std::vector<T> diffs(ref_points.size() - 1);
    std::transform(std::next(ref_points.begin()), ref_points.end(),
                   ref_points.begin(), diffs.begin(),
                   Distance<T>()
                  );
    std::vector<T> t(ref_points.size());
    T L = std::reduce(diffs.begin(), diffs.end());
    T sum_dist = 0;
    for (size_t i = 0; i < ref_points.size() - 1; ++i) {
        t[i] = sum_dist / L;
        sum_dist += diffs[i];
    }
    t[ref_points.size() - 1] = 1;
    std::vector<T> dt(ref_points.size() - 1);
    std::transform(std::next(t.begin()), t.end(), t.begin(), dt.begin(), std::minus<T>());
    T min_dparam = *std::min_element(dt.begin(), dt.end());
    return {min_dparam, t};
}

//! A method for calculation of the spline knots and storing the data inside the spline
template <class T, int Degree>
std::vector<T> SplineSolver<T, Degree>::CalculateKnots(const std::vector<T> &t) {
    std::vector<T> knots(t.size() + Degree + 1);
    for (size_t i = 0; i < Degree + 1; ++i) {
        knots[i] = 0;
        knots[knots.size() - 1 - i] = 1;
    }
    T avg = 0;
    for (size_t i = 1; i < Degree + 1; ++i) {
        avg += t[i];
    }
    for (size_t i = 1; i < t.size() - Degree; ++i) {
        knots[i + Degree] = avg / Degree;
        avg -= t[i];
        avg += t[i + Degree];
    }
    return knots;
}

//! A method for calculation of one line of N matrix coefficients
template <class T, int Degree>
std::vector<T> SplineSolver<T, Degree>::CalculateCoeffsMatrix(const std::vector<T> &t,
                                                              const std::vector<T> &knots) const {
    std::vector<T> N(t.size() * t.size());
    for (size_t i = 0; i < t.size(); ++i) {
        auto n = CalculateCoeffs(t[i], knots);
        std::copy(n.begin(), n.end(), N.begin() + i * t.size());
    }
    return N;
}

//! A method for calculation of N matrix coefficients
template <class T, int Degree>
std::vector<T> SplineSolver<T, Degree>::CalculateCoeffs(T t, const std::vector<T> &knots) const {
    size_t points_count = knots.size() - Degree - 1;
    std::vector<T> N(points_count, 0);
    // Special cases
    if (t == knots[0]) {
        N[0] = 1;
        return N;
    }
    if (t == knots[knots.size() - 1]) {
        N[points_count - 1] = 1;
        return N;
    }
    // Find knots interval
    size_t k = 0;
    for (size_t i = 0; i < knots.size() - 1; ++i) {
        if (t >= knots[i] && t < knots[i + 1]) {
            k = i;
            break;
        }
    }
    // Calculate coeffs
    N[k] = 1;
    for (size_t d = 1; d <= Degree; ++d) {
        N[k - d] = (knots[k + 1] - t) * N[(k - d) + 1] / (knots[k + 1] - knots[k - d + 1]);
        for (size_t i = k - d + 1; i < k; ++ i) {
            N[i] = (t - knots[i]) * N[i] / (knots[i + d] - knots[i]) +
                   (knots[i + d + 1] - t) * N[i + 1] / (knots[i + d + 1] - knots[i + 1]);
        }
        N[k] = (t - knots[k]) * N[k] / (knots[k + d] - knots[k]);
    }
    return N;
}

//! A method for calculation of P and Q coefficients of a spline
template <class T, int Degree>
std::tuple<std::vector<T>, std::vector<T>>
SplineSolver<T, Degree>::Solve(const std::vector<T> &N,
                               const std::vector<T> &knots,
                               const std::vector<std::vector<T>> &ref_points) {
    Eigen::MatrixX<T> n;
    size_t points_count = ref_points.size();
    n.resize(points_count, points_count);
    for (size_t i = 0; i < points_count; ++i) {
        for (size_t j = 0; j < points_count; ++j) {
            n(i,j) = N[i * points_count + j];
        }
    }
    Eigen::FullPivLU<Eigen::Ref<Eigen::MatrixX<T>>> lu(n);
    Eigen::VectorX<T> d, p;
    d.resize(points_count);
    p.resize(points_count);
    std::vector<T> P(points_count * Spline<T, Degree>::Dim),
        Q((points_count - 1) * Spline<T, Degree>::Dim);
    for (size_t i = 0; i < Spline<T, Degree>::Dim; ++i) {
        for (size_t j = 0; j < points_count; ++j) {
            d(j) = ref_points[j][i];
        }
        p = lu.solve(d);
        for (size_t j = 0; j < points_count; ++j) {
            P[i * points_count + j] = p(j);
            if (j > 0) {
                Q[i * (points_count - 1) + j - 1] =
                    (p(j) - p(j - 1)) * Degree / (knots[j + Degree] - knots[j]);
            }
        }
    }
    return {P, Q};
}

template class SplineSolver<double, 3>;

} // namespace spline
