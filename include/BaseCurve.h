#pragma once
#include <cuda/std/array>
#include <cuda_runtime_api.h>

namespace spline {

//! An interface class for a basic function
//!
//! A function is parametric differentiable function
//!
template <class T, class Curve>
class IBaseFunction {
public:
    //! A method for returning a value of a parametric function
    //!
    //!\param arg - an argument of the function
    //!\return a vector of the coordinates of a point representing the value of a function
    //!
    __host__ __device__ cuda::std::array<T, 2> operator()(T arg) const;

    //! A method for returning a derivative of a parametric function
    //!
    //!\param arg - an argument of a derivative of a function
    //!\return a vector of the coordinates of a point representing the value of a derivative a function
    //!
    __host__ __device__ cuda::std::array<T, 2> Derivative(T arg) const;
};

//! An interface class for all curves
//!
//! The curves are 2D parametric functions
//!
template <class T, class Curve>
class IBaseCurve: public IBaseFunction<T, IBaseCurve<T, Curve>> {
public:
    //! Dimention of the curve
    static constexpr size_t Dim = 2;

    //! A default constructor
    IBaseCurve() = default;

    //! A constructor with predefined minimum parameter distance
    //!
    //! \param min_dparam - a value of predefined minimum parameter distance
    //!
    IBaseCurve(T min_dparam): min_dparam_(min_dparam) {}

    //! A method for returning the point on a curve depending on a parameter
    //!
    //! \param t - a parameter of a function
    //! \return a vector of coordinates of a point representing the value of a curve
    //!
    __host__ __device__ cuda::std::array<T, 2> operator()(T t) const;

    //! A method for returning a derivative of a curve depending on a parameter
    //!
    //! \param t - a parameter of a derivative of a function
    //! \return a vector of coordinates of a point representing the value of a derivative of a curve
    //!
    __host__ __device__ cuda::std::array<T, 2> Derivative(T t) const;

    //! A method to get the minimum parameter distance between consecutive special points
    //!
    //! The special points start a changing of curvature. For example for a B-spline this are reference points.
    //! \return a minimum parameter distance between consecutive special points
    //!
    __host__ __device__ T GetMinDParam() const {
        return min_dparam_;
    }

    //! A method to copy internal data from host to device memory
    //!
    //! \return the resulting curve
    //!
    const Curve &Sync();

protected:
    //! A value of predefined minimum parameter distance
    T min_dparam_;
};

} //! namespace spline
