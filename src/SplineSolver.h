#pragma once
#include <vector>
#include <tuple>

namespace spline {

//! The class for a 2D parametric B-spline solving from the vector of reference points
template <class T, int Degree>
class SplineSolver {
public:
    //! A method for calculation of all spline data
    //!
    //! \param ref_points - a vector of reference points
    //! \return a tuple of min_dparam, knots, P coeffs and Q coeffs
    //!
    std::tuple<T, std::vector<T>, std::vector<T>, std::vector<T>>
        SolveSpline(const std::vector<std::vector<T>> &ref_points);

    //! A method for calculation of t parameters and minimum parameters distance
    //! from spline reference points
    //!
    //! \param ref_points - a vector of reference points
    //!
    std::tuple<T, std::vector<T>> CalculateParams(const std::vector<std::vector<T>> &ref_points);

    //! A method for calculation of the spline knots and storing the data inside the spline
    std::vector<T> CalculateKnots(const std::vector<T> &t);

    //! A method for calculation of one line of N matrix coefficients
    //!
    //! \param t - a value of the spline t parameter
    //! \param knots - a vector of spline knots
    //! \return a vector of N matrix coefficients
    //!
    std::vector<T> CalculateCoeffs(T t, const std::vector<T> &knots) const;

    //! A method for calculation of N matrix coefficients
    //!
    //! \param t - a vector of spline t parameter
    //! \param knots - a vector of spline knots
    //! \return a vector of full N matrix coefficients
    //!
    std::vector<T> CalculateCoeffsMatrix(const std::vector<T> &t,
                                         const std::vector<T> &knots) const;

    //! A method for calculation of P and Q coefficients of a spline
    //!
    //! The method calculates P coefficients for spline points calculation
    //! and Q coefficients for spline derivative construction
    //!
    //! \param N - full spline N matrix
    //! \param knots - a vector of spline knots
    //! \param ref_points - a vector of spline reference points
    //!
    std::tuple<std::vector<T>, std::vector<T> > Solve(const std::vector<T> &N,
                                                      const std::vector<T> &knots,
                                                      const std::vector<std::vector<T>> &ref_points);

};

} //! namespace spline
