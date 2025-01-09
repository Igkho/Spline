#include "SearchInitializer.h"
#include "ComplexFunctions.h"
#include <map>
#include <thread>
#include <cmath>

namespace spline {

template <class T, class Curve, class Other>
std::vector<std::vector<T>> SearchInitializer<T, Curve, Other>::GetSearchGrid(const T ratio) const {
    T dt0 = curve0_.GetMinDParam() * ratio;
    T dt1 = curve1_.GetMinDParam() * ratio;
    dt0 = std::min((T)MAX_DPARAM, std::max((T)MIN_DPARAM, dt0));
    dt1 = std::min((T)MAX_DPARAM, std::max((T)MIN_DPARAM, dt1));
    size_t count0 = (size_t)std::ceil((T)1 / dt0), count1 = (size_t)std::ceil((T)1 / dt1);
    std::vector<std::vector<T>> result((count0 + 1) * (count1 + 1));
    size_t nthreads = std::thread::hardware_concurrency();
#pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i <= count0; ++i) {
        for (int j = 0; j <= count1; ++j) {
            result[i * (count1 + 1) + j] = {
                std::min((T)1, std::max((T)0, (T)i * dt0)),
                std::min((T)1, std::max((T)0, (T)j * dt1))
            };
        }
    }
    return result;
}

template <class T, class Curve, class Other>
std::vector<std::vector<T> >
SearchInitializer<T, Curve, Other>::GetSearchPoints(T search_ratio, T select_ratio) const {
    auto sg = GetSearchGrid(search_ratio);
    std::map<T, std::vector<T>> f_sorted;
    L2Norm<T, Curve, Other> l2n({curve0_, curve1_});
    for (const auto &p : sg) {
        f_sorted[l2n(p[0], p[1])[0]] = p;
    }
    size_t num_tries = (select_ratio == -1 ? f_sorted.size() :
                        (T)1 / (std::min(curve0_.GetMinDParam(),
                                         curve1_.GetMinDParam())) * select_ratio);
    std::vector<std::vector<T>> results(num_tries, {(T)0, (T)0});
    for (auto [i, it] = std::tuple{0, f_sorted.begin()}; i < num_tries && it != f_sorted.end(); ++it, ++i) {
        results[i] = it->second;
    }
    return results;
}

template class SearchInitializer<double, SplineD<double, 3>, SplineD<double, 3>>;
template class SearchInitializer<double, Ellipse<double>, SplineD<double, 3>>;
template class SearchInitializer<double, SplineD<double, 3>, Ellipse<double>>;
template class SearchInitializer<double, Ellipse<double>, Ellipse<double>>;

} // namespace spline
