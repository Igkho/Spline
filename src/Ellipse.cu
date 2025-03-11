#include "Ellipse.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cmath>

namespace spline {

//! A method for returning the point on an ellipse depending on a parameter
template <class T>
__host__ __device__ cuda::std::array<T, 2> Ellipse<T>::operator()(T t) const {
    T arg = (t > (T).5 ? t - (T)1 : t);
    T r0, r1;
    r0 = scale_[0] * std::cos(2.0 * M_PI * arg) + shift_[0];
    r1 = scale_[1] * std::sin(2.0 * M_PI * arg) + shift_[1];
    return {r0, r1};
}

//! A method for returning the value of a derivative of an ellipse depending on a parameter
template <class T>
__host__ __device__ cuda::std::array<T, 2> Ellipse<T>::Derivative(T t) const {
    T arg = (t > (T).5 ? t - 1 : t);
    T r0, r1;
    r0 = - 2 * M_PI * scale_[0] * std::sin(2.0 * M_PI * arg);
    r1 = 2 * M_PI * scale_[1] * std::cos(2.0 * M_PI * arg);
    return {r0, r1};
}

//! A method to fill internal cuda data buffer with device memory data
template <class T>
void Ellipse<T>::FillCudaBuffer() {
    if (cuda_buffer_.size() == 0) {
        Sync();
        CurveCuda<T> curve;
        curve.type = CurveCuda<T>::CurveType::ellipse;
        curve.min_dparam = this->GetMinDParam();
        curve.ellipse.shift = GetShift();
        curve.ellipse.scale = GetScale();
        std::vector<CurveCuda<T>> curve_cpu{curve};
        cuda_buffer_ = std::move(Block<CurveCuda<T>>(curve_cpu));
    }
}

//! A method to fill internal cuda data buffer with device memory data (float type)
template <class T>
void Ellipse<T>::FillCudaBufferFloat() {
    if (cuda_buffer_float_.size() == 0) {
        Sync();
        CurveCuda<float> curve;
        curve.type = CurveCuda<float>::CurveType::ellipse;
        curve.min_dparam = this->GetMinDParam();
        curve.ellipse.shift = {(float)GetShift()[0], (float)GetShift()[1]};
        curve.ellipse.scale = {(float)GetScale()[0], (float)GetScale()[1]};
        std::vector<CurveCuda<float>> curve_cpu{curve};
        cuda_buffer_float_ = std::move(Block<CurveCuda<float>>(curve_cpu));
    }
}

template class Ellipse<double>;

} // namespace spline
