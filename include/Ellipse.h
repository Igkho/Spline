#pragma once
#include "BaseCurve.h"
#include "CurveCuda.h"
#include "Block.h"
#include <cuda/std/array>
#include <cuda_runtime_api.h>

namespace spline {

//! The class for a 2D parametric ellipse
template <class T>
class Ellipse: public IBaseCurve<T, Ellipse<T>> {
public:
    //! A default minimum parameter distance for an ellipse
    static constexpr T MIN_DPARAM = 0.25; //! i.e. 4 search points on an ellipse

    //! A default constructor
    Ellipse() = default;
    ~Ellipse() = default;

    //! A constructor of an ellipse
    //!
    //! \param scale - a vector of coordinates of a scale value
    //! \param shift - a vector of coordinates of a shift value
    //!
    Ellipse(const cuda::std::array<T, 2> &scale, const cuda::std::array<T, 2> &shift):
        IBaseCurve<T, Ellipse<T>>(MIN_DPARAM), scale_(scale), shift_(shift) {}

    //! A method for returning the point on an ellipse depending on a parameter
    //!
    //! \param t - a parameter value
    //! \return a vector of coordinates of a point on an ellipse
    //!
    __host__ __device__ cuda::std::array<T, 2> operator ()(T t) const;

    //! A method for returning the value of a derivative of an ellipse depending on a parameter
    //!
    //! \param t - a parameter value
    //! \return a vector of derivative values in a point on an ellipse
    //!
    __host__ __device__ cuda::std::array<T, 2> Derivative(T t) const;

    //! A dummy method to copy internal data from host to device memory
    //!
    //! For ellipses does nothing.
    //! \return the resulting ellipse
    //!
    __host__ __device__ const Ellipse<T> &Sync() {
        return *this;
    }

    //! A getter for scale values
    __host__ __device__ const cuda::std::array<T, 2> &GetScale() const {
        return scale_;
    }

    //! A getter for shift values
    __host__ __device__ const cuda::std::array<T, 2> &GetShift() const {
        return shift_;
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
    //! An internal cuda data buffer
    Block<CurveCuda<T>> cuda_buffer_;
    //! An internal cuda data buffer (float type)
    Block<CurveCuda<float>> cuda_buffer_float_;
    //! Scale and shift values
    cuda::std::array<T, 2> scale_, shift_;
};

} //! namespace spline
