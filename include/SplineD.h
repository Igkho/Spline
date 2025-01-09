#pragma once
#include "BaseCurve.h"
#include "CurveCuda.h"
#include <cuda/std/array>
#include "Block.h"
#include "DualBlock.h"
#include <cuda_runtime_api.h>

namespace spline {

//! The class for a 2D parametric B-spline without the derivative
template <class T, int Degree>
class Spline: public IBaseCurve<T, Spline<T, Degree>> {
public:
    //! A default constructor
    Spline() = default;

    //! A default destructor
    ~Spline() = default;

    //! A constructor on the basis of known knots and spline coefficients
    //!
    //! \param min_dparam - a minimum parameter distance value
    //! \param knots - a vector of coordinates of knots
    //! \param P - a vector of spline parameters for points calculation
    //!
    Spline(T min_dparam, const std::vector<T> &knots, const std::vector<T> &P);

    //! A getter for a number of reference points
    __host__ __device__ size_t GetPointsCount() const {
        return knots_.size() - Degree - 1;
    }

    //! A getter for a vector of knots of a spline
    __host__ __device__ const DualBlock<T> &GetKnots() const {
        return knots_;
    }

    //! A getter for a vector of P parameters of a spline
    __host__ __device__ const DualBlock<T> &GetPCoefficients() const {
        return P_;
    }

    //! Method for returning the point on a spline depending on the parameter
    //!
    //! \param t - a parameter value
    //! \return a vector of coordinates of a point on a spline
    //!
    __host__ __device__ cuda::std::array<T, 2> operator ()(T t) const;

    //! A method for returning the value of a derivative of a spline
    //!
    //! \param t - a parameter value
    //! \return a dumb zero vector (this is the class for the spline without derivative)
    //!
    __host__ __device__ cuda::std::array<T, 2> Derivative(T t) const;

    //! A method to copy internal data from host to device memory
    //!
    //! \return the resulting spline
    //!
    const Spline<T, Degree> &Sync() {
        knots_.sync();
        P_.sync();
        return *this;
    }

protected:
    //! A DualBlock of spline knots
    DualBlock<T> knots_;
    //! A DualBlock of spline P coefficients
    DualBlock<T> P_;
};

//! The class for a B-spline with derivative
template <class T, int Degree>
class SplineD: public IBaseCurve<T, SplineD<T, Degree>> {
public:
    //! A default constructor
    SplineD() = default;

    //! A default destructor
    ~SplineD() = default;

    //! A constructor on the basis of reference points
    //!
    //! \param ref_points - a vector of coordinates of referense points for a spline
    //!
    SplineD(const std::vector<std::vector<T>> &ref_points);

    //! A constructor on the basis of spline and derivative spline
    //!
    //! \param spline - the spline itself
    //! \param derivative - the derivative spline
    //!
    SplineD(const Spline<T, Degree> &spline, const Spline<T, Degree - 1> derivative) :
        spline_(spline), derivative_(derivative) {}

    //! Method for returning the point on a spline depending on the parameter
    //!
    //! \param t - a parameter value
    //! \return a vector of coordinates of a point on a spline
    //!
    __host__ __device__ cuda::std::array<T, 2> operator ()(T t) const {
        return spline_(t);
    }

    //! A method for returning the value of a derivative of a spline
    //!
    //! \param t - a parameter value
    //! \return a vector of derivative values in a point on a spline
    //!
    __host__ __device__ cuda::std::array<T, 2> Derivative(T t) const {
        return derivative_(t);
    }

    //! A method to get the minimum parameter distance between consecutive special points
    //!
    //! The special points start a changing of curvature. For example for a B-spline this are reference points.
    //! \return a minimum parameter distance between consecutive special points
    //!
    __host__ __device__ T GetMinDParam() const {
        return spline_.GetMinDParam();
    }

    //! A getter for a spline
    __host__ __device__ const Spline<T, Degree> &GetSpline() const {
        return spline_;
    }

    //! A getter for a derivative spline
    __host__ __device__ const Spline<T, Degree - 1> &GetDerivative() const {
        return derivative_;
    }

    //! A method to copy internal data from host to device memory
    //!
    //! \return the resulting spline
    //!
    const SplineD<T, Degree> &Sync() {
        spline_.Sync();
        derivative_.Sync();
        return *this;
    }

    //! A method to fill internal cuda data buffer with device memory data
    void FillCudaBuffer();

    //! A method to fill internal cuda data buffer with device memory data (float type)
    void FillCudaBufferFloat();

    //! A getter for internal cuda data buffer
    const CurveCuda<T> *GetCudaBuffer() {
        return cuda_buffer_.data();
    }

    //! A getter for internal cuda data buffer (float type)
    const CurveCuda<float> *GetCudaBufferFloat() {
        return cuda_buffer_float_.data();
    }

private:
    //! A spline
    Spline<T, Degree> spline_;
    //! A derivative spline
    Spline<T, Degree - 1> derivative_;
    //! A float type spline (cuda use only)
    Spline<float, Degree> spline_float_;
    //! A derivative float type spline (cuda use only)
    Spline<float, Degree - 1> derivative_float_;
    //! A cuda data buffer
    Block<CurveCuda<T>> cuda_buffer_;
    //! A cuda data buffer(float type)
    Block<CurveCuda<float>> cuda_buffer_float_;
};

} //! namespace spline
