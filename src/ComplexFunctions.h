#pragma once
#include "Spline.h"
#include <cuda/std/array>
#include <stdexcept>
#include "helpers.h"

namespace spline {

// Interface class for complex function
template <class T, class Curve0, class Curve1, class CFunc>
class IComplexFunction {
public:
    __host__ __device__ IComplexFunction(const Curve0 &func0, const Curve1 &func1) :
        func0_(func0), func1_(func1) {}

    // A method for returning the value of a function
    __host__ __device__ cuda::std::array<T, 2> operator()(T arg0, T arg1) const;

    // A method for returning the derivative of a function
    __host__ __device__ cuda::std::array<T, 4> Derivative(T arg0, T arg1) const;
protected:
    const Curve0 &func0_;
    const Curve1 &func1_;
};

// A class for difference of two functions
template <class T, class Curve0, class Curve1>
class Difference : public IComplexFunction<T, Curve0, Curve1, Difference<T, Curve0, Curve1>> {
public:
    __host__ __device__ Difference(const Curve0 &func0, const Curve1 &func1) :
        IComplexFunction<T, Curve0, Curve1, Difference<T, Curve0, Curve1>>(func0, func1) {}

    // A method for difference calculation
    __host__ __device__ cuda::std::array<T, 2> operator()(T arg0, T arg1) const;

    // A method for difference derivative calculation
    __host__ __device__ cuda::std::array<T, 4> Derivative(T arg0, T arg1) const;
};

// A class for L2 norm
template <class T, class Curve0, class Curve1>
class L2Norm: public IComplexFunction<T, Curve0, Curve1, L2Norm<T, Curve0, Curve1>> {
public:
    __host__ __device__ L2Norm(const Curve0 &func0, const Curve1 &func1) :
        IComplexFunction<T, Curve0, Curve1, L2Norm<T, Curve0, Curve1>>(func0, func1) {}

    // A method for L2Norm calculation
    __host__ __device__ cuda::std::array<T, 2> operator()(T arg0, T arg1) const;

    // A method for L2Norm derivative calculation
    __host__ __device__ cuda::std::array<T, 4> Derivative(T arg0, T arg1) const;
};

} // namespace spline
