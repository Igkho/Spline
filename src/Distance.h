#pragma once
#include <cmath>
#include <numeric>
#include <functional>

namespace spline {

//! A class for two points distance calculation
template <class T>
struct Distance: std::function<T(const std::vector<T> &, const std::vector<T> &)> {
    T operator()(const std::vector<T> &a, const std::vector<T> &b) {
        return std::sqrt(std::transform_reduce(a.begin(),
                                               a.end(),
                                               b.begin(),
                                               (T)0,
                                               std::plus<T>(),
                                               [](const T &lhs, const T &rhs){
                                                   return (lhs - rhs) * (lhs - rhs);
                                               }));
    }
};

} //! namespace spline
