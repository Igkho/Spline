#include "ComplexFunctions.h"

namespace spline {

// The method for complex function value calculation (CRT pattern)
template <class T, class Curve0, class Curve1, class CFunc>
cuda::std::array<T, 2> IComplexFunction<T, Curve0, Curve1, CFunc>::operator()(T arg0, T arg1) const {
    return static_cast<const CFunc *>(this)->operator()(arg0, arg1);
}

// The method for complex function derivative calculation (CRT pattern)
template <class T, class Curve0, class Curve1, class CFunc>
cuda::std::array<T, 4> IComplexFunction<T, Curve0, Curve1, CFunc>::Derivative(T arg0, T arg1) const {
    return static_cast<const CFunc *>(this)->Derivative(arg0, arg1);
}

// The method for difference calculation
template <class T, class Curve0, class Curve1>
cuda::std::array<T, 2> Difference<T, Curve0, Curve1>::operator()(T arg0, T arg1) const {
    auto f0 = (this->func0_)(arg0);
    auto f1 = (this->func1_)(arg1);
    return {f0[0] - f1[0], f0[1] - f1[1]};
}

// The method for difference derivative calculation
template <class T, class Curve0, class Curve1>
cuda::std::array<T, 4> Difference<T, Curve0, Curve1>::Derivative(T arg0, T arg1) const {
    auto d0 = (this->func0_).Derivative(arg0);
    auto d1 = (this->func1_).Derivative(arg1);
    return {d0[0], d0[1], -d1[0], -d1[1]};
}

// The method for L2Norm calculation
template <class T, class Curve0, class Curve1>
cuda::std::array<T, 2> L2Norm<T, Curve0, Curve1>::operator()(T arg0, T arg1) const {
    auto f0 = (this->func0_)(arg0);
    auto f1 = (this->func1_)(arg1);
    T delta = f0[0] - f1[0];
    T norm = delta * delta;
    delta = f0[1] - f1[1];
    norm += delta * delta;
    return {norm, norm};
}

// The method for L2Norm derivative calculation
template <class T, class Curve0, class Curve1>
cuda::std::array<T, 4> L2Norm<T, Curve0, Curve1>::Derivative(T arg0, T arg1) const {
    auto d0 = this->func0_.Derivative(arg0);
    auto d1 = this->func1_.Derivative(arg1);
    auto f0 = (this->func0_)(arg0);
    auto f1 = (this->func1_)(arg1);
    T delta = f0[0] - f1[0];
    T v0 = 2 * delta * d0[0];
    T v1 = - 2 * delta * d1[0];
    delta = f0[1] - f1[1];
    v0 += 2 * delta * d0[1];
    v1 -= 2 * delta * d1[1];
    return {v0, v1, v0, v1};
}

template class Difference<double, SplineD<double, 3>, SplineD<double, 3>>;
template class Difference<double, Ellipse<double>, SplineD<double, 3>>;
template class Difference<double, SplineD<double, 3>, Ellipse<double>>;
template class Difference<double, Ellipse<double>, Ellipse<double>>;

template class L2Norm<double, SplineD<double, 3>, SplineD<double, 3>>;
template class L2Norm<double, Ellipse<double>, SplineD<double, 3>>;
template class L2Norm<double, SplineD<double, 3>, Ellipse<double>>;
template class L2Norm<double, Ellipse<double>, Ellipse<double>>;


} // namespace spline
