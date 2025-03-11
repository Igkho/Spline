#include "Intersector.h"
#include "IntersectorKernels.h"
#include "Distance.h"
#include "SearchInitializer.h"
#include "Optimizers.h"
#include <vector>
#include <stdexcept>
#include <set>
#include <map>
#include <thread>
#include <algorithm>

namespace spline {

//! A method to find intersections of two curves on the basis of the Newton-Raphson method
template <class T, class Curve0, class Curve1>
std::vector<std::vector<T>>
Intersector<T, Curve0, Curve1>::Intersect(T epsilon,
                                          size_t max_iters) {
    if (curve0_.Dim != curve1_.Dim) {
        throw std::runtime_error("The curves should have equal dimensions");
    }
    auto results = SearchInitializer<T, Curve0, Curve1>(curve0_, curve1_).
            GetSearchPoints(INTERSECT_SEARCH_GRID_RATIO);
    size_t nthreads = std::thread::hardware_concurrency();
#pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < results.size(); ++i) {
        NROptimizer<T, Curve0, Curve1> opt(curve0_, curve1_, epsilon);
        auto r = opt.Optimize(results[i][0], results[i][1], max_iters);
        results[i] = {r[0], r[1]};
    }
    results.push_back({(T)0, (T)0});
    results.push_back({(T)0, (T)1});
    results.push_back({(T)1, (T)0});
    results.push_back({(T)1, (T)1});
    std::set<std::vector<T>> unique;
    L2Norm<T, Curve0, Curve1> l2n(curve0_, curve1_);
    for (const auto &p : results) {
        if (l2n(p[0], p[1])[0] < epsilon * epsilon) {
            unique.insert(p);
        }
    }
    results.clear();
    std::vector<T> last;
    for (const auto &v : unique) {
        if (last.size() == 0 || Distance<T>{}(last, v) > epsilon) {
            results.push_back(v);
            last = v;
        }
    }
    return results;
}

//! The method to find closest points of two curves on the basis of the
//! RMSProp optimization or brute force search
template <class T, class Curve0, class Curve1>
std::vector<std::vector<T>>
Intersector<T, Curve0, Curve1>::Closest(ClosestOptAlgorithm alg,
                                        T epsilon,
                                        size_t max_iters,
                                        T alpha,
                                        T beta,
                                        size_t p_split,
                                        T p_step
                                        ) {
    if (curve0_.Dim != curve1_.Dim) {
        throw std::runtime_error("The curves should have equal dimensions");
    }
    std::vector<std::vector<T>> results;
    if (alg == ClosestOptAlgorithm::BruteForce_cuda) {
        CudaGetSearchPoints<T, Curve0, Curve1>(results_cuda_,
                                               curve0_,
                                               curve1_,
                                               CLOSEST_SEARCH_GRID_RATIO,
                                               SELECT_RATIO,
                                               SearchInitializer<T, Curve0, Curve1>::MIN_DPARAM,
                                               SearchInitializer<T, Curve0, Curve1>::MAX_DPARAM
                                              );
        results = CudaOptimize<T, Curve0, Curve1>(results_cuda_,
                                                  deltas_,
                                                  curve0_,
                                                  curve1_,
                                                  max_iters,
                                                  epsilon,
                                                  p_split,
                                                  p_step,
                                                  CLOSEST_SEARCH_GRID_RATIO
                                                 );
    } else {
        results = SearchInitializer<T, Curve0, Curve1>(curve0_, curve1_).
            GetSearchPoints(CLOSEST_SEARCH_GRID_RATIO);
        size_t nthreads = std::thread::hardware_concurrency();
#pragma omp parallel for num_threads(nthreads)
        for (int i = 0; i < results.size(); ++i) {
            if (alg == ClosestOptAlgorithm::BruteForce_cpu) {
                BruteForceOptimizer<T, Curve0, Curve1> opt(curve0_, curve1_, p_split, p_step, epsilon);
                opt.Optimize(results[i][0], results[i][1], CLOSEST_SEARCH_GRID_RATIO);
            } else {
                RMSPropOptimizer<T, Curve0, Curve1> opt(curve0_, curve1_, alpha, beta, epsilon);
                auto r = opt.Optimize(results[i][0], results[i][1], max_iters,
                                      CLOSEST_SEARCH_GRID_RATIO);
                results[i] = {r[0], r[1]};
            }
        }
    }
    results.push_back({(T)0, (T)0});
    results.push_back({(T)0, (T)1});
    results.push_back({(T)1, (T)0});
    results.push_back({(T)1, (T)1});

    // Sort the argument pairs by calculated distance
    std::vector<std::pair<T, std::vector<T>>> f_sorted;
    L2Norm<T, Curve0, Curve1> l2n(curve0_, curve1_);
    for (const auto &p : results) {
        f_sorted.push_back({l2n(p[0], p[1])[0], p});
    }
    std::sort(f_sorted.begin(), f_sorted.end(), [](const std::pair<T, std::vector<T>> &lhs,
                                                   const std::pair<T, std::vector<T>> &rhs) {
                                                    return lhs.first < rhs.first;
                                                });
    T min_dist = f_sorted.begin()->first;
    std::vector<T> sum(2, (T)0);
    size_t count = 0;
    auto min_p = f_sorted.begin()->second;
    // Calculate the average of successful attempts
    for (const auto [dist, coords] : f_sorted) {
        if (std::abs(min_dist - dist) > epsilon * min_dist) {
            break;
        }
        sum[0] += coords[0];
        sum[1] += coords[1];
        count++;
    }
    sum[0] /= count;
    sum[1] /= count;
    return {sum};
}

template class Intersector<double, SplineD<double, 3>, SplineD<double, 3>>;
template class Intersector<double, Ellipse<double>, Ellipse<double>>;
template class Intersector<double, Ellipse<double>, SplineD<double, 3>>;
template class Intersector<double, SplineD<double, 3>, Ellipse<double>>;

} // namespace spline
