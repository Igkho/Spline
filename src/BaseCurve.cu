#include "BaseCurve.h"
#include "Ellipse.h"
#include "SplineD.h"
#include <cuda/std/array>

namespace spline {

//! A method for returning a value of a parametric function (CRT pattern)
template <class T, class Curve>
cuda::std::array<T, 2> IBaseFunction<T, Curve>::operator()(T arg) const {
    return static_cast<const Curve *>(this)->operator()(arg);
}

//! A method for returning a derivative of a parametric function (CRT pattern)
template <class T, class Curve>
cuda::std::array<T, 2> IBaseFunction<T, Curve>::Derivative(T arg) const {
    return static_cast<const Curve *>(this)->Derivative(arg);
}

//! A method for returning the point on a curve depending on a parameter (CRT pattern)
template <class T, class Curve>
cuda::std::array<T, 2> IBaseCurve<T, Curve>::operator()(T arg) const {
    return static_cast<const Curve *>(this)->operator()(arg);
}

//! A method for returning a derivative of a curve depending on a parameter (CRT pattern)
template <class T, class Curve>
cuda::std::array<T, 2> IBaseCurve<T, Curve>::Derivative(T arg) const {
    return static_cast<const Curve *>(this)->Derivative(arg);
}

//! A method to copy internal data from host to device memory (CRT pattern)
template <class T, class Curve>
const Curve &IBaseCurve<T, Curve>::Sync() {
    return static_cast<Curve *>(this)->Sync();
}

template class IBaseCurve<double, Ellipse<double>>;
template class IBaseCurve<double, Spline<double, 3>>;
template class IBaseCurve<double, Spline<double, 2>>;
template class IBaseCurve<double, SplineD<double, 3>>;

} // namespace spline
