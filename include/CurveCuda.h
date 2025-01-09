#pragma once
#include <cuda/std/array>

namespace spline {

//! A cuda data buffer class for splines
template <class T>
struct SplineCuda {
    //! A device memory pointer to knots data
    const T *knots;
    //! A length of knots data
    int knots_size;
    //! A device memory pointer to P coefficients data (spline calculation)
    const T *P;
    //! A device memory pointer to Q coefficients data (spline derivative calculation)
    const T *Q;
};

//! A cuda data buffer class for ellipses
template <class T>
struct EllipseCuda {
    //! Cuda array for ellipse scale data
    cuda::std::array<T, 2> scale;
    //! Cuda array for ellipse shift data
    cuda::std::array<T, 2> shift;
};

//! A cuda data buffer class for curves (splines or ellipses)
template <class T>
struct CurveCuda {
    //! A curve type enum
    enum class CurveType {ellipse, spline};
    //! A curve type value
    CurveType type;
    //! A minimum parameter distance between consecutive special points
    T min_dparam;
    //! A union cuda data buffer for splines or ellipses (depending on the type)
    union {
        SplineCuda<T> spline;
        EllipseCuda<T> ellipse;
    };
};

} //! namespace spline
