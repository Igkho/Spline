#include "Optimizers.h"

namespace spline {

template <class T, class Curve0, class Curve1, class Opt>
cuda::std::array<T, 2>
IOptimizer<T, Curve0, Curve1, Opt>::Optimize(T arg0,
                                             T arg1,
                                             size_t max_iters,
                                             T ratio
                                            ) {
    return static_cast<Opt *>(this)->Optimize(arg0, arg1, max_iters, ratio);
}

template <class T, class Curve0, class Curve1, class Opt>
bool
IOptimizer<T, Curve0, Curve1, Opt>::OptimizeStep(T &arg0,
                                                 T &arg1,
                                                 cuda::std::array<T, 4> &values) {
    return static_cast<Opt *>(this)->OptimizeStep(arg0, arg1, values);
}

template <class T, class Curve0, class Curve1>
cuda::std::array<T, 2> NROptimizer<T, Curve0, Curve1>::Optimize(T arg0,
                                                                T arg1,
                                                                size_t max_iters,
                                                                T ratio
                                                               ) {
    auto v = diff_(arg0, arg1);
    T arg_opt0 = arg0, arg_opt1 = arg1;
    cuda::std::array<T, 4> values{v[0], v[1], v[0], v[1]};
    for (size_t i = 0; i < max_iters; ++i) {
        if (OptimizeStep(arg_opt0, arg_opt1, values)) {
            return {arg_opt0, arg_opt1};
        }
        if (std::abs(arg_opt0 - arg0) >= this->func0_.GetMinDParam() * ratio ||
            std::abs(arg_opt1 - arg1) >= this->func1_.GetMinDParam() * ratio) {
            return {arg0, arg1};
        }
    }
    return {arg0, arg1};
}

template <class T, class Curve0, class Curve1>
bool NROptimizer<T, Curve0, Curve1>::OptimizeStep(T &arg0,
                                                  T &arg1,
                                                  cuda::std::array<T, 4> &values) {
    auto df = diff_.Derivative(arg0, arg1);
    T a = df[0], b = df[2], c = df[1], d = df[3];
    T inv_det = (T)1 / (a * d - b * c);
    arg0 -= (d * values[0] - b * values[1]) * inv_det;
    arg0 = std::max((T)0, std::min((T)1, arg0));
    arg1 -= (a * values[1] - c * values[0]) * inv_det;
    arg1 = std::max((T)0, std::min((T)1, arg1));
    auto vn = diff_(arg0, arg1);
    if (std::abs(vn[0] - values[0]) < (T)0.5 * epsilon_ * std::max((T)1, std::abs(vn[0])) &&
        std::abs(vn[1] - values[1]) < (T)0.5 * epsilon_ * std::max((T)1, std::abs(vn[1]))) {
        return true;
    }
    values = {vn[0], vn[1], vn[0], vn[1]};
    return false;
}

template <class T, class Curve0, class Curve1>
cuda::std::array<T, 2> RMSPropOptimizer<T, Curve0, Curve1>::Optimize(T arg0,
                                                                     T arg1,
                                                                     size_t max_iters,
                                                                     T ratio) {
    v0_ = v1_ = T(0);
    auto v0 = this->func0_(arg0);
    auto v1 = this->func1_(arg1);
    cuda::std::array<T, 4> values{v0[0], v0[1], v1[0], v1[1]};
    T arg_opt0 = arg0, arg_opt1 = arg1;
    for (size_t i = 0; i < max_iters; ++i) {
        if (OptimizeStep(arg_opt0, arg_opt1, values)) {
            return {arg_opt0, arg_opt1};
        }
        if (std::abs(arg_opt0 - arg0) >= this->func0_.GetMinDParam() * std::max(0.1, ratio) ||
            std::abs(arg_opt1 - arg1) >= this->func1_.GetMinDParam() * std::max(0.1, ratio)) {
            return {arg0, arg1};
        }
    }
    return {arg0, arg1};
}

template <class T, class Curve0, class Curve1>
bool RMSPropOptimizer<T, Curve0, Curve1>::OptimizeStep(T &arg0,
                                                       T &arg1,
                                                       cuda::std::array<T, 4> &values) {
    auto df = l2n_.Derivative(arg0, arg1);
    v0_ = beta_ * v0_ + (1 - beta_) * df[0] * df[0];
    v1_ = beta_ * v1_ + (1 - beta_) * df[1] * df[1];
    arg0 -= alpha_ * df[0] / (std::sqrt(v0_));
    arg0 = std::max((T)0, std::min((T)1, arg0));
    arg1 -= alpha_ * df[1] / (std::sqrt(v1_));
    arg1 = std::max((T)0, std::min((T)1, arg1));
    auto vn0 = this->func0_(arg0);
    auto vn1 = this->func1_(arg1);
    T dv00 = vn0[0] - values[0];
    T dv01 = vn0[1] - values[1];
    T dv10 = vn1[0] - values[2];
    T dv11 = vn1[1] - values[3];
    T dV = std::sqrt(dv00 * dv00 + dv01 * dv01 + dv10 * dv10 + dv11 * dv11);
    if (dV < (T)0.5 * epsilon_ * l2n_(arg0, arg1)[0]) {
        return true;
    }
    values = {vn0[0], vn0[1], vn1[0], vn1[1]};
    return false;
}

template <class T, class Curve0, class Curve1>
void BruteForceOptimizer<T, Curve0, Curve1>::Optimize(T &arg0, T &arg1, T ratio) {
    auto delta0 = this->func0_.GetMinDParam() * ratio;
    auto delta1 = this->func1_.GetMinDParam() * ratio;
    size_t max_iters = std::ceil(std::log((T)1.0/(epsilon_)) / std::log((T)power_step_));
    for (size_t i = 0; i < max_iters; ++i) {
        OptimizeStep(arg0, arg1, delta0, delta1);
    }
}

template <class T, class Curve0, class Curve1>
void BruteForceOptimizer<T, Curve0, Curve1>::OptimizeStep(T &arg0,
                                                          T &arg1,
                                                          T &delta0,
                                                          T &delta1) {
    T min0 = arg0, min1 = arg1, min_val = l2n_(arg0, arg1)[0];
    for (size_t i = 0; i <= power_split_; ++i) {
        for (size_t j = 0; j <= power_split_; ++j) {
            T argn0 = arg0 - delta0 + (T)2 * i * delta0 / power_split_;
            argn0 = std::max((T)0, std::min((T)1, argn0));
            T argn1 = arg1 - delta1 + (T)2 * j * delta1 / power_split_;
            argn1 = std::max((T)0, std::min((T)1, argn1));
            T val = l2n_(argn0, argn1)[0];
            if (val < min_val) {
                min_val = val;
                min0 = argn0;
                min1 = argn1;
            }
        }
    }
    delta0 /= power_step_;
    delta1 /= power_step_;
    arg0 = min0;
    arg1 = min1;
}

template class NROptimizer<double, SplineD<double, 3>, SplineD<double, 3>>;
template class NROptimizer<double, Ellipse<double>, SplineD<double, 3>>;
template class NROptimizer<double, SplineD<double, 3>, Ellipse<double>>;
template class NROptimizer<double, Ellipse<double>, Ellipse<double>>;

template class RMSPropOptimizer<double, SplineD<double, 3>, SplineD<double, 3>>;
template class RMSPropOptimizer<double, Ellipse<double>, SplineD<double, 3>>;
template class RMSPropOptimizer<double, SplineD<double, 3>, Ellipse<double>>;
template class RMSPropOptimizer<double, Ellipse<double>, Ellipse<double>>;

template class BruteForceOptimizer<double, SplineD<double, 3>, SplineD<double, 3>>;
template class BruteForceOptimizer<double, Ellipse<double>, SplineD<double, 3>>;
template class BruteForceOptimizer<double, SplineD<double, 3>, Ellipse<double>>;
template class BruteForceOptimizer<double, Ellipse<double>, Ellipse<double>>;

} // namespace spline
