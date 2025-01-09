#pragma once
#include "Ellipse.h"
#include "SplineD.h"
#include "Block.h"
#include <vector>

namespace spline {


template <class T, class Curve0, class Curve1>
std::vector<std::vector<T>>  CudaOptimize(Block<T> &results,
                                          Curve0 &c0,
                                          Curve1 &c1,
                                          size_t max_iters,
                                          T epsilon,
                                          size_t power_split = 1,
                                          T power_step = (T)1,
                                          T ratio = (T)1
                                         );

extern template std::vector<std::vector<double>> CudaOptimize<double, SplineD<double, 3>, SplineD<double, 3>>(
    Block<double> &,
    SplineD<double, 3> &,
    SplineD<double, 3> &,
    size_t,
    double,
    size_t,
    double,
    double
   );

extern template std::vector<std::vector<double>> CudaOptimize<double, Ellipse<double>, SplineD<double, 3>>(
    Block<double> &,
    Ellipse<double> &,
    SplineD<double, 3> &,
    size_t,
    double,
    size_t,
    double,
    double
   );

extern template std::vector<std::vector<double>> CudaOptimize<double, SplineD<double, 3>, Ellipse<double>>(
    Block<double> &,
    SplineD<double, 3> &,
    Ellipse<double> &,
    size_t,
    double,
    size_t,
    double,
    double
   );

extern template std::vector<std::vector<double>> CudaOptimize<double, Ellipse<double>, Ellipse<double>>(
    Block<double> &,
    Ellipse<double> &,
    Ellipse<double> &,
    size_t,
    double,
    size_t,
    double,
    double
   );

template <class T, class Curve0, class Curve1>
void CudaGetSearchPoints(Block<T> &,
                         Curve0 &c0,
                         Curve1 &c1,
                         T min_dparam,
                         T max_dparam,
                         T select_ratio,
                         T ratio = (T)1
                        );

extern template void
CudaGetSearchPoints<double, Ellipse<double>, Ellipse<double>>(
    Block<double> &,
    Ellipse<double> &,
    Ellipse<double> &,
    double,
    double,
    double,
    double
   );

extern template void
CudaGetSearchPoints<double, Ellipse<double>, SplineD<double, 3>>(
    Block<double> &,
    Ellipse<double> &,
    SplineD<double, 3> &,
    double,
    double,
    double,
    double
   );

extern template void
CudaGetSearchPoints<double, SplineD<double, 3>, Ellipse<double>>(
    Block<double> &,
    SplineD<double, 3> &,
    Ellipse<double> &,
    double,
    double,
    double,
    double
   );

extern template void
CudaGetSearchPoints<double, SplineD<double, 3>, SplineD<double, 3>>(
    Block<double> &,
    SplineD<double, 3> &,
    SplineD<double, 3> &,
    double,
    double,
    double,
    double
   );

} // namespace spline
