#include "Spline_impl.h"
#include <iostream>
#include <iomanip>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <stdexcept>
#include <map>
#include <set>
#include <thread>
#include "Eigen/Eigen"

namespace spline {

template <class T>
std::vector<std::vector<T>> SearchInitializer<T>::GetSearchGrid(const T ratio) const {
    std::vector<std::vector<T>> result;
    T dt0 = curves_[0]->GetMinDParam() * ratio;
    T dt1 = curves_[1]->GetMinDParam() * ratio;
    dt0 = std::min((T)MAX_DPARAM, std::max((T)MIN_DPARAM, dt0));
    dt1 = std::min((T)MAX_DPARAM, std::max((T)MIN_DPARAM, dt1));
    for (size_t i = 0; i <= (T)1 / dt0; ++i) {
        for (size_t j = 0; j <= (T)1 / dt1; ++j) {
            result.push_back({(T)i * dt0, (T)j * dt1});
        }
    }
    return result;
}

template <class T>
std::tuple<std::vector<std::vector<T>>, std::vector<std::vector<T>>>
    SearchInitializer<T>::GetSearchPointsAndInitResults(T search_ratio) const {
    auto sg = GetSearchGrid(search_ratio);
    std::map<T, std::vector<T>> f_sorted;
    L2Norm<T> l2n({curves_[0], curves_[1]});
    for (const auto &p : sg) {
        f_sorted[l2n(p)[0]] = p;
    }
    size_t num_tries = (T)1 / std::min(curves_[0]->GetMinDParam(),
                                       curves_[1]->GetMinDParam()) * 2;
    std::vector<std::vector<T>> results(num_tries + 1, {(T)0, (T)0});
    results.push_back({(T)0, (T)1});
    results.push_back({(T)1, (T)0});
    results.push_back({(T)1, (T)1});
    std::vector<std::vector<T>> try_points(num_tries, {(T)0, (T)0});
    for (auto [i, it] = std::tuple{0, f_sorted.begin()}; i < num_tries && it != f_sorted.end(); ++it, ++i) {
        results[i] = it->second;
        try_points[i] = it->second;
    }
    return {try_points, results};
}

template <class T>
std::vector<T> NROptimizer<T>::Optimize(const std::vector<T> &args,
                                        size_t max_iters) {
    std::vector<std::vector<T>> values = {diff_(args)};
    std::vector<T> args_opt = args;
    for (size_t i = 0; i < max_iters; ++i) {
        if (OptimizeStep(args_opt, values)) {
            return args_opt;
        }
    }
    return args;
}

template <class T>
bool NROptimizer<T>::OptimizeStep(std::vector<T> &args,
                                  std::vector<std::vector<T>> &values) {
    auto df = diff_.Derivative(args);
    Eigen::MatrixX<T> J;
    J.resize(2, 2);
    J(0, 0) = df[0][0];
    J(1, 0) = df[0][1];
    J(0, 1) = df[1][0];
    J(1, 1) = df[1][1];
    Eigen::VectorX<T> Fe, X;
    Fe.resize(2);
    X.resize(2);
    Fe(0) = values[0][0];
    Fe(1) = values[0][1];
    Eigen::FullPivLU<Eigen::Ref<Eigen::MatrixX<T>>> lu(J);
    X = lu.solve(Fe);
    args[0] -= X(0);
    args[0] = std::max((T)0, std::min((T)1, args[0])),
    args[1] -= X(1);
    args[1] = std::max((T)0, std::min((T)1, args[1]));
    std::vector<std::vector<T>> values_new = {diff_(args)};
    if (std::abs(values_new[0][0] - values[0][0]) <
            (T)0.5 * epsilon_ * std::max((T)1, std::abs(values_new[0][0])) &&
        std::abs(values_new[0][1] - values[0][1]) <
            (T)0.5 * epsilon_ * std::max((T)1, std::abs(values_new[0][1]))) {
        return true;
    }
    values = values_new;
    return false;
}

template <class T>
std::vector<std::vector<T>> BasicCurve<T>::Intersect(const BasicCurve<T> &other,
                                                     T epsilon,
                                                     size_t max_iters) const {
    if (Dim != other.Dim) {
        throw std::runtime_error("The other curve should have same dimensions");
    }
    // Initialize the search points
    SearchInitializer<T> si({this, &other});
    auto [points, results] = si.GetSearchPointsAndInitResults();
    // Find the points candidates
    size_t nthreads = std::thread::hardware_concurrency();
#pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < points.size(); ++i) {
        NROptimizer<T> opt({this, &other}, epsilon);
        results[i] = opt.Optimize(points[i], max_iters);
    }
    // Select the unique points
    std::set<std::vector<T>> unique;
    L2Norm<T> l2n({this, &other});
    for (const auto &p : results) {
        if (l2n(p)[0] < epsilon * epsilon) {
            unique.insert(p);
        }
    }
    results.clear();
    std::vector<T> last;
    for (const auto &v : unique) {
        if (last.size() == 0 || Distance(last, v) > epsilon) {
            results.push_back(v);
            last = v;
        }
    }
    return results;
}

template <class T>
std::vector<T> RMSPropOptimizer<T>::Optimize(const std::vector<T> &args,
                                             size_t max_iters) {
    v_ = std::vector<T>(2, 0);
    std::vector<std::vector<T>> values = {(*this->funcs_[0])(args[0]),
                                          (*this->funcs_[1])(args[1])};
    std::vector<T> args_opt = args;
    for (size_t i = 0; i < max_iters; ++i) {
        if (OptimizeStep(args_opt, values)) {
            return args_opt;
        }
    }
    return args;
}

template <class T>
bool RMSPropOptimizer<T>::OptimizeStep(std::vector<T> &args,
                                       std::vector<std::vector<T>> &values) {
    auto df = l2n_.Derivative(args)[0];
    v_[0] = beta_ * v_[0] + (1 - beta_) * df[0] * df[0];
    v_[1] = beta_ * v_[1] + (1 - beta_) * df[1] * df[1];
    args[0] -= alpha_ * df[0] / (std::sqrt(v_[0]));
    args[0] = std::max((T)0, std::min((T)1, args[0])),
    args[1] -= alpha_ * df[1] / (std::sqrt(v_[1]));
    args[1] = std::max((T)0, std::min((T)1, args[1]));
    std::vector<std::vector<T>> values_new = {(*this->funcs_[0])(args[0]),
                                              (*this->funcs_[1])(args[1])};
    T dv00 = values_new[0][0] - values[0][0];
    T dv01 = values_new[0][1] - values[0][1];
    T dv10 = values_new[1][0] - values[1][0];
    T dv11 = values_new[1][1] - values[1][1];
    T dV = std::sqrt(dv00 * dv00 + dv01 * dv01 + dv10 * dv10 + dv11 * dv11);
    if (dV < (T)0.5 * epsilon_ * l2n_(args)[0]) {
        return true;
    }
    values = values_new;
    return false;
}

template <class T>
std::vector<std::vector<T>> BasicCurve<T>::Closest(const BasicCurve<T> &other,
                                                   T epsilon,
                                                   size_t max_iters,
                                                   T alpha,
                                                   T beta) const {
    if (Dim != other.Dim) {
        throw std::runtime_error("Curves should have same dimensions");
    }
    // Initialize the search points
    SearchInitializer<T> si({this, &other});
    auto [points, results] = si.GetSearchPointsAndInitResults(CLOSEST_SEARCH_GRID_RATIO);
    // Find the points candidates
    size_t nthreads = std::thread::hardware_concurrency();
#pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < points.size(); ++i) {
        RMSPropOptimizer<T> opt({this, &other}, alpha, beta, epsilon);
        results[i] = opt.Optimize(points[i], max_iters);
    }
    // Sort the argument pairs by calculated distance
    std::map<T, std::vector<T>> f_sorted;
    L2Norm<T> l2n({this, &other});
    for (const auto &p : results) {
        f_sorted[l2n(p)[0]] = p;
    }
    T min_dist = f_sorted.begin()->first;
    std::vector<T> sum(2, (T)0);
    size_t count = 0;
    // Calculate the average of successful attempts
    for (const auto [dist, coords] : f_sorted) {
        if (std::abs(min_dist - dist) > epsilon * min_dist) {
            break;
        }
        sum[0] += coords[0];
        sum[1] += coords[1];
        count ++;
    }
    sum[0] /= count;
    sum[1] /= count;
    return {sum};
}

template <class T>
T BasicCurve<T>::Distance(const std::vector<T> &a, const std::vector<T> &b) const {
    T sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

template <class T>
std::vector<T> Ellipse<T>::operator()(T t) const {
    std::vector<T> result(this->Dim);
    result[0] = scale_[0] * std::cos(2.0 * M_PI * t) + shift_[0];
    result[1] = scale_[1] * std::sin(2.0 * M_PI * t) + shift_[1];
    return result;
}

template <class T>
std::vector<T> Ellipse<T>::Derivative(T t) const {
    std::vector<T> result(this->Dim);
    result[0] = - 2 * M_PI * scale_[0] * std::sin(2.0 * M_PI * t);
    result[1] = 2 * M_PI * scale_[1] * std::cos(2.0 * M_PI * t);
    return result;
}

template <class T, int Degree>
Spline<T, Degree>::Spline(size_t points_count, const std::vector<T> &knots, const std::vector<T> &P):
    points_count_(points_count), knots_(knots), P_(P) {
    if (points_count_ <= Degree) {
        throw std::runtime_error("Direct points count should be higher than spline degree");
    }
    if (knots_.size() != points_count_ + Degree + 1) {
        throw std::runtime_error("Wrong knots count");
    }
    if (P_.size() != points_count_ * this->Dim) {
        throw std::runtime_error("Wrong points count");
    }
}

// The class for a B-spline
template <class T, int Degree>
Spline<T, Degree>::Spline(const std::vector<std::vector<T>> &ref_points) {
    points_count_ = ref_points.size();
    if (points_count_ <= Degree) {
        throw std::runtime_error("Points count should be higher than spline degree");
    }
    // Check points validity
    for (const auto &p : ref_points) {
        if (p.size() != this->Dim) {
            throw std::runtime_error("All points sizes should be equal to 2");
        }
    }
    // Calculate chord lengths
    std::vector<T> diffs = CalculateDiffs(ref_points);

    // Calculate chord length parameters
    CalculateParams(diffs);

    // Calculate knots
    CalculateKnots();

    // Calculate coefficients
    std::vector<T> N = CalculateCoeffsMatrix();

    // Solve the spline
    Solve(N, ref_points);
}

template <class T, int Degree>
std::vector<T> Spline<T, Degree>::CalculateDiffs(const std::vector<std::vector<T>> &ref_points) const {
    std::vector<T> diffs(points_count_ - 1);
    std::transform(std::next(ref_points.begin()), ref_points.end(),
                   ref_points.begin(), diffs.begin(),
                   [this](const std::vector<T> &a, const std::vector<T> &b) {
                       return this->Distance(a, b);
                   });
    return diffs;
}

template <class T, int Degree>
std::vector<T> Spline<T, Degree>::CalculateCoeffs(T t) const {
    std::vector<T> N(points_count_, 0);
    // Special cases
    if (t == knots_[0]) {
        N[0] = 1;
        return N;
    }
    if (t == knots_[knots_.size() - 1]) {
        N[points_count_ - 1] = 1;
        return N;
    }
    // Find knots interval
    size_t k = 0;
    for (size_t i = 0; i < knots_.size() - 1; ++i) {
        if (t >= knots_[i] && t < knots_[i + 1]) {
            k = i;
            break;
        }
    }
    // Calculate coeffs
    N[k] = 1;
    for (size_t d = 1; d <= Degree; ++d) {
        N[k - d] = (knots_[k + 1] - t) * N[(k - d) + 1] / (knots_[k + 1] - knots_[k - d + 1]);
        for (size_t i = k - d + 1; i < k; ++ i) {
            N[i] = (t - knots_[i]) * N[i] / (knots_[i + d] - knots_[i]) +
                   (knots_[i + d + 1] - t) * N[i + 1] / (knots_[i + d + 1] - knots_[i + 1]);
        }
        N[k] = (t - knots_[k]) * N[k] / (knots_[k + d] - knots_[k]);
    }
    return N;
}

template <class T, int Degree>
void Spline<T, Degree>::CalculateParams(const std::vector<T> diffs) {
    t_.resize(points_count_);
    T L = std::reduce(diffs.begin(), diffs.end());
    T sum_dist = 0;
    for (size_t i = 0; i < points_count_ - 1; ++i) {
        t_[i] = sum_dist / L;
        sum_dist += diffs[i];
    }
    t_[points_count_ - 1] = 1;
    std::vector<T> dt(points_count_ - 1);
    std::transform(std::next(t_.begin()), t_.end(), t_.begin(), dt.begin(), std::minus<T>());
    this->min_dparam_ = *std::min_element(dt.begin(), dt.end());
}

template <class T, int Degree>
std::vector<T> Spline<T, Degree>::CalculateCoeffsMatrix() const {
    std::vector<T> N(points_count_ * points_count_);
    for (size_t i = 0; i < points_count_; ++i) {
        auto n = CalculateCoeffs(t_[i]);
        std::copy(n.begin(), n.end(), N.begin() + i * points_count_);
    }
    return N;
}

template <class T, int Degree>
std::vector<T> Spline<T, Degree>::operator()(T t) const {
    std::vector<T> result(this->Dim);
    auto n = CalculateCoeffs(t);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = std::transform_reduce(n.begin(), n.end(), P_.begin() + i * n.size(), (T)0);
    }
    return result;
}

template <class T, int Degree>
void Spline<T, Degree>::Solve(const std::vector<T> &n,
                              const std::vector<std::vector<T>> &ref_points) {
    Eigen::MatrixX<T> N;
    N.resize(points_count_, points_count_);
    for (size_t i = 0; i < points_count_; ++i) {
        for (size_t j = 0; j < points_count_; ++j) {
            N(i,j) = n[i * points_count_ + j];
        }
    }
    Eigen::FullPivLU<Eigen::Ref<Eigen::MatrixX<T>>> lu(N);
    Eigen::VectorX<T> D, P;
    D.resize(points_count_);
    P.resize(points_count_);
    P_.resize(points_count_ * this->Dim);
    Q_.resize((points_count_ - 1) * this->Dim);
    for (size_t i = 0; i < this->Dim; ++i) {
        for (size_t j = 0; j < points_count_; ++j) {
            D(j) = ref_points[j][i];
        }
        P = lu.solve(D);
        for (size_t j = 0; j < points_count_; ++j) {
            P_[i * points_count_ + j] = P(j);
            if (j > 0) {
                Q_[i * (points_count_ - 1) + j - 1] =
                    (P(j) - P(j - 1)) * Degree / (knots_[j + Degree] - knots_[j]);
            }
        }
    }
}

template <class T, int Degree>
void Spline<T, Degree>::CalculateKnots() {
    knots_.resize(points_count_ + Degree + 1);
    for (size_t i = 0; i < Degree + 1; ++i) {
        knots_[i] = 0;
        knots_[knots_.size() - 1 - i] = 1;
    }
    T avg = 0;
    for (size_t i = 1; i < Degree + 1; ++i) {
        avg += t_[i];
    }
    for (size_t i = 1; i < t_.size() - Degree; ++i) {
        knots_[i + Degree] = avg / Degree;
        avg -= t_[i];
        avg += t_[i + Degree];
    }
}

template class BasicCurve<double>;
template class Ellipse<double>;
template class Spline<double, 2>;
template class SplineD<double, 3>;

} // namespace spline
