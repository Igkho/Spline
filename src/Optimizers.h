#pragma once
// #include "Spline.h"
#include "ComplexFunctions.h"
#include <cuda/std/array>
#include <cuda_runtime_api.h>

namespace spline {

// Interface class for optimizers
template <class T, class Curve0, class Curve1, class Opt>
class IOptimizer {
public:
    __host__ __device__ IOptimizer(const Curve0 &func0, const Curve1 &func1) :
        func0_(func0), func1_(func1) {}

    // The pure virtual method for iterative optimization
    __host__ __device__ cuda::std::array<T, 2> Optimize(T arg0,
                                                        T arg1,
                                                        size_t max_iters,
                                                        T ratio = (T)1
                                                       );

protected:
    const Curve0 &func0_;
    const Curve1 &func1_;

    // The pure virtual method for one optimization step
    __host__ __device__ bool OptimizeStep(T &arg0, T &arg1, cuda::std::array<T, 4> &values);
};

// Newton-Raphson optimizer class
template <class T, class Curve0, class Curve1>
class NROptimizer: public IOptimizer<T, Curve0, Curve1, NROptimizer<T, Curve0, Curve1>> {
public:
    __host__ __device__ NROptimizer(const Curve0 &func0, const Curve1 &func1, T epsilon) :
        IOptimizer<T, Curve0, Curve1, NROptimizer<T, Curve0, Curve1>>(func0, func1),
        diff_(func0, func1), epsilon_(epsilon) {}

    __host__ __device__ cuda::std::array<T, 2> Optimize(T arg0,
                                                        T arg1,
                                                        size_t max_iters,
                                                        T ratio = (T)1
                                                       );

protected:
    const Difference<T, Curve0, Curve1> diff_;
    const T epsilon_;

    __host__ __device__ bool OptimizeStep(T &arg0, T &arg1, cuda::std::array<T, 4> &values);
};

// RMSProp optimizer class
template <class T, class Curve0, class Curve1>
class RMSPropOptimizer: public IOptimizer<T, Curve0, Curve1, RMSPropOptimizer<T, Curve0, Curve1>> {
public:
    __host__ __device__ RMSPropOptimizer(const Curve0 &func0, const Curve1 &func1,
                     T alpha, T beta, T epsilon) :
        IOptimizer<T, Curve0, Curve1, RMSPropOptimizer<T, Curve0, Curve1>>(func0, func1),
        l2n_(func0, func1), alpha_(alpha), beta_(beta), epsilon_(epsilon) {}

    __host__ __device__ cuda::std::array<T, 2> Optimize(T arg0,
                                                        T arg1,
                                                        size_t max_iters,
                                                        T ratio = (T)1
                                                       );

protected:
    T v0_, v1_;
    const L2Norm<T, Curve0, Curve1> l2n_;
    T alpha_, beta_, epsilon_;

    __host__ __device__ bool OptimizeStep(T &arg0, T &arg1, cuda::std::array<T, 4> &values);
};

// BruteForce optimizer class
template <class T, class Curve0, class Curve1>
class BruteForceOptimizer: public IOptimizer<T, Curve0, Curve1,
                                              BruteForceOptimizer<T, Curve0, Curve1>> {
public:
    __host__ __device__ BruteForceOptimizer(const Curve0 &func0,
                                            const Curve1 &func1,
                                            size_t power_split,
                                            T power_step,
                                            T epsilon
                                           ) :
        IOptimizer<T, Curve0, Curve1, BruteForceOptimizer<T, Curve0, Curve1>>(func0, func1),
        l2n_(func0, func1), epsilon_(epsilon), power_step_(power_step), power_split_(power_split) {}

    void Optimize(T &arg0, T &arg1, T ratio = (T)1);

protected:
    const L2Norm<T, Curve0, Curve1> l2n_;
    const T epsilon_, power_step_;
    const size_t power_split_;

    __host__ __device__ void OptimizeStep(T &arg0, T &arg1, T &delta0, T &delta1);
};

} // namespace spline
