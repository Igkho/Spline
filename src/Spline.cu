#include "SplineD.h"
#include "SplineSolver.h"
#include <cuda/std/array>
#include <stdexcept>

namespace spline {

//! A constructor on the basis of known knots and spline coefficients
template <class T, int Degree>
Spline<T, Degree>::Spline(T min_dparam, const std::vector<T> &knots, const std::vector<T> &P):
    knots_(knots), P_(P) {
    this->min_dparam_ = min_dparam;
    if (GetPointsCount() <= Degree) {
        throw std::runtime_error("Points count should be higher than spline degree");
    }
    if (P_.size() != GetPointsCount() * this->Dim) {
        throw std::runtime_error("P size and knots size do not match");
    }
}

//! Method for returning the point on a spline depending on the parameter
template <class T, int Degree>
cuda::std::array<T, 2> Spline<T, Degree>::operator()(T t) const {
    // Special cases
    if (t == knots_[0]) {
        return {*P_.begin(), *(P_.begin() + GetPointsCount())};
    }
    if (t == knots_[knots_.size() - 1]) {
        return {*(P_.begin() + GetPointsCount() - 1), *(P_.end() - 1)};
    }
    cuda::std::array<T, Degree + 1> N;
    for (size_t i = 0; i <= Degree; ++i) {
        N[i] = (T)0;
    }
    size_t k = 0;
    for (size_t i = 0; i < knots_.size() - 1; ++i) {
        if (t >= knots_[i] && t < knots_[i + 1]) {
            k = i;
            break;
        }
    }
    // Calculate coeffs
    N[Degree] = (T)1;
    for (size_t d = 1; d <= Degree; ++d) {
        N[Degree - d] = (knots_[k + 1] - t) * N[(Degree - d) + 1] /
                        (knots_[k + 1] - knots_[k - d + 1]);
        for (size_t i = k - d + 1; i < k; ++i) {
            N[i - k + Degree] = (t - knots_[i]) * N[i - k + Degree] / (knots_[i + d] - knots_[i]) +
                   (knots_[i + d + 1] - t) * N[i - k + Degree + 1] / (knots_[i + d + 1] - knots_[i + 1]);
        }
        N[Degree] = (t - knots_[k]) * N[Degree] / (knots_[k + d] - knots_[k]);
    }
    T r0 = (T)0, r1 = (T)0;
    for (size_t i = 0; i <= Degree; ++i) {
        r0 += N[i] * P_[k - Degree + i];
        r1 += N[i] * P_[k - Degree + i + GetPointsCount()];
    }
    return {r0, r1};
}

//! A method for returning the value of a derivative of a spline (zeros for the spline without derivative)
template <class T, int Degree>
cuda::std::array<T, 2> Spline<T, Degree>::Derivative(T t) const {
    return {(T)0, (T)0};
}

//! A constructor on the basis of reference points
template <class T, int Degree>
SplineD<T, Degree>::SplineD(const std::vector<std::vector<T>> &ref_points) {
    auto ss = SplineSolver<T, Degree>();
    auto [min_dparam, knots, P, Q] = ss.SolveSpline(ref_points);
    spline_ = Spline<T, Degree>(min_dparam, knots, P),
    derivative_ = Spline<T, Degree - 1>(min_dparam, {knots.begin() + 1, knots.end() - 1}, Q);
}

//! A method to fill internal cuda data buffer with device memory data
template <class T, int Degree>
void SplineD<T, Degree>::FillCudaBuffer() {
    if (cuda_buffer_.size() == 0) {
        Sync();
        CurveCuda<T> curve;
        curve.type = CurveCuda<T>::CurveType::spline;
        curve.min_dparam = spline_.GetMinDParam();
        curve.spline.knots = spline_.GetKnots().GetBlockCuda().data();
        curve.spline.knots_size = spline_.GetKnots().GetBlockCuda().size();
        curve.spline.P = spline_.GetPCoefficients().GetBlockCuda().data();
        curve.spline.Q = derivative_.GetPCoefficients().GetBlockCuda().data();
        std::vector<CurveCuda<T>> curve_cpu{curve};
        cuda_buffer_ = std::move(Block<CurveCuda<T>>(curve_cpu));
    }
}

//! A method to fill internal cuda data buffer with device memory data (float type)
template <class T, int Degree>
void SplineD<T, Degree>::FillCudaBufferFloat() {
    if (cuda_buffer_float_.size() == 0) {
        Sync();
        CurveCuda<float> curve;
        curve.type = CurveCuda<float>::CurveType::spline;
        curve.min_dparam = spline_.GetMinDParam();
        std::vector<float> knots_float(spline_.GetKnots().begin(), spline_.GetKnots().end());
        std::vector<float> P_float(spline_.GetPCoefficients().begin(),
                                   spline_.GetPCoefficients().end());
        std::vector<float> d_knots_float(knots_float.begin() + 1, knots_float.end() - 1);
        std::vector<float> Q_float(derivative_.GetPCoefficients().begin(),
                                   derivative_.GetPCoefficients().end());
        float min_dparam = (float)spline_.GetMinDParam();
        spline_float_ = std::move(Spline<float, 3>(min_dparam, knots_float, P_float));
        derivative_float_ = std::move(Spline<float, 2>(min_dparam, d_knots_float, Q_float));
        spline_float_.Sync();
        derivative_float_.Sync();
        curve.spline.knots = spline_float_.GetKnots().GetBlockCuda().data();
        curve.spline.knots_size = spline_float_.GetKnots().GetBlockCuda().size();
        curve.spline.P = spline_float_.GetPCoefficients().GetBlockCuda().data();
        curve.spline.Q = derivative_float_.GetPCoefficients().GetBlockCuda().data();
        std::vector<CurveCuda<float>> curve_cpu{curve};
        cuda_buffer_float_ = std::move(Block<CurveCuda<float>>(curve_cpu));
    }
}

template class Spline<double, 2>;
template class Spline<double, 3>;
template class SplineD<double, 3>;

template class Spline<float, 2>;
template class Spline<float, 3>;

} // namespace spline
