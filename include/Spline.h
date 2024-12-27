#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

namespace spline {

//! An interface class for a basic function
//!
//! A function is parametric differentiable function
//!
template <class T>
class IBasicFunction {
public:
    //! A pure virtual method for returning a value of a parametric function
    //!
    //!\param arg - an argument of the function
    //!\return a vector of the coordinates of a point representing the value of a function
    //!
    virtual std::vector<T> operator()(T arg) const = 0;

    //! A pure virtual method for returning a derivative of a parametric function
    //!
    //!\param arg - an argument of a derivative of a function
    //!\return a vector of the coordinates of a point representing the value of a derivative a function
    //!
    virtual std::vector<T> Derivative(T arg) const = 0;
};

//! An interface class for all curves
//!
//! The curves are 2D parametric functions
//!
template <class T>
class BasicCurve: public IBasicFunction<T> {
public:
    //! Dimention of the curve
    static constexpr size_t Dim = 2;
    //! A dafault relative precision of points calculation
    static constexpr T EPSILON = 1e-10;
    //! A default maximum number of iterations while finding curves intersections
    static constexpr size_t MAX_INTERSECTION_ITERS = 20;
    //! A default maximum number of iterations while finding curves closest points
    static constexpr size_t MAX_CLOSEST_ITERS = 5000;
    //! A default alpha value for RMSProp optimizer (finding closest points)
    static constexpr T CLOSEST_ALPHA = 0.0001;
    //! A default beta value for RMSProp optimizer (finding closest points)
    static constexpr T CLOSEST_BETA = 0.9999;
    //! A default ratio for selecting initial points (finding intersection points)
    static constexpr size_t INTERSECT_SELECT_RATIO = 3;
    //! A default ratio for init search grid calculation (finding closest points)
    static constexpr T CLOSEST_SEARCH_GRID_RATIO = 0.1;

    //! A default constructor
    BasicCurve() = default;

    //! A constructor with predefined minimum parameter distance
    //!
    //! \param min_dparam - a value of predefined minimum parameter distance
    //!
    BasicCurve(T min_dparam): min_dparam_(min_dparam) {}

    //! A pure virtual method for returning the point on a curve depending on a parameter
    //!
    //! \param t - a parameter of a function
    //! \return a vector of coordinates of a point representing the value of a curve
    //!
    virtual std::vector<T> operator()(T t) const = 0;

    //! A pure virtual method for returning a derivative of a curve depending on a parameter
    //!
    //! \param t - a parameter of a derivative of a function
    //! \return a vector of coordinates of a point representing the value of a derivative of a curve
    //!
    virtual std::vector<T> Derivative(T t) const = 0;

    //! A method to get the minimum parameter distance between consecutive special points
    //!
    //! The special points start a changing of curvature. For example for a B-spline this are reference points.
    //! \return a minimum parameter distance between consecutive special points
    //!
    T GetMinDParam() const {
        return min_dparam_;
    }

    //! A method to find intersections of two curves on the basis of the Newton-Raphson method
    //!
    //! \param other - another curve to find intersections with this one
    //! \param epsilon - relative precision of points calculation
    //! \param max_iters - maximum number of iterations while finding curves intersections for each point on a search grid
    //! \return a vector of points (each point is a vector of coordinates) of found curves intersections
    //!
    std::vector<std::vector<T>> Intersect(const BasicCurve<T> &other,
                                          T epsilon = EPSILON,
                                          size_t max_iters = MAX_INTERSECTION_ITERS) const;

    //! The method to find closest points of two curves on the basis of the RMSProp optimization
    //!
    //! \param other - another curve to find closest points to
    //! \param epsilon - relative precision of points calculation
    //! \param max_iters - maximum number of iterations while finding curves closest points for each point on a search grid
    //! \param alpha - alpha value for RMSProp optimizer
    //! \param beta - beta value for RMSProp optimizer
    //! \return a vector of two points (each point is a vector of coordinates) which are found to be the closest
    //!
    std::vector<std::vector<T>> Closest(const BasicCurve<T> &other,
                                        T epsilon = EPSILON,
                                        size_t max_iters = MAX_CLOSEST_ITERS,
                                        T alpha = CLOSEST_ALPHA,
                                        T beta = CLOSEST_BETA) const;

protected:
    //! A value of predefined minimum parameter distance
    T min_dparam_;

    //! Method for calculation the distance between two points
    //!
    //! \param a - a vector of coordinates of the first point
    //! \param b - a vector of  coordinates of the second point
    //! \return a value of the distance between the points
    //!
    T Distance(const std::vector<T> &a, const std::vector<T> &b) const;

};

//! The class for a 2D parametric ellipse
template <class T>
class Ellipse: public BasicCurve<T> {
public:
    //! A default minimum parameter distance for an ellipse
    static constexpr T MIN_DPARAM = 0.25; //! i.e. 4 search points on an ellipse

    //! A constructor of an ellipse
    //!
    //! \param scale - a vector of coordinates of a scale value
    //! \param shift - a vector of coordinates of a shift value
    //!
    Ellipse(std::vector<T> scale, std::vector<T> shift):
        BasicCurve<T>(MIN_DPARAM), scale_(scale), shift_(shift) {
        if (scale_.size() != this->Dim || shift_.size() != this->Dim) {
            throw std::runtime_error("All input vector sizes should be equal to 2");
        }
    }

    //! A method for returning the point on an ellipse depending on a parameter
    //!
    //! \param t - a parameter value
    //! \return a vector of coordinates of a point on an ellipse
    //!
    virtual std::vector<T> operator()(T t) const override;

    //! A method for returning the value of a derivative of an ellipse depending on a parameter
    //!
    //! \param t - a parameter value
    //! \return a vector of derivative values in a point on an ellipse
    //!
    virtual std::vector<T> Derivative(T t) const override;

private:
    std::vector<T> scale_, shift_;
};

//! The class for a 2D parametric B-spline without the derivative
template <class T, int Degree>
class Spline: public BasicCurve<T> {
public:
    //! A default constructor
    Spline() = default;

    //! A constructor on the basis of reference points
    //!
    //! \param ref_points - a vector of coordinates of referense points for a spline
    //!
    Spline(const std::vector<std::vector<T>> &ref_points);

    //! A constructor for a derivative spline
    //!
    //! \param points_count - a number of reference points
    //! \param knots - a vector of coordinates of knots
    //! \param P - a vector of spline parameters for points calculation
    //!
    Spline(size_t points_count, const std::vector<T> &knots, const std::vector<T> &P);

    //! A getter for a number of reference points
    size_t GetPointsCount() const {
        return points_count_;
    }

    //! A getter for a vector of t parameters of a spline
    const std::vector<T> &GetTParameters() const {
        return t_;
    }

    //! A getter for a vector of knots of a spline
    const std::vector<T> &GetKnots() const {
        return knots_;
    }

    //! A getter for a vector of P parameters of a spline
    const std::vector<T> &GetPCoefficients() const {
        return P_;
    }

    //! A getter for a vector of Q parameters of a spline
    const std::vector<T> &GetQCoefficients() const {
        return Q_;
    }

    //! Method for returning the point on a spline depending on the parameter
    //!
    //! \param t - a parameter value
    //! \return a vector of coordinates of a point on a spline
    //!
    virtual std::vector<T> operator()(T t) const override;

    //! A method for returning the value of a derivative of a spline
    //!
    //! \param t - a parameter value
    //! \return a dumb zero vector (this is the class for the spline without derivative)
    //!
    virtual std::vector<T> Derivative(T t) const override {
        std::vector<T> result(this->Dim, 0);
        return result;
    }

protected:

    //! A method for calculation of differences for spline reference points
    //!
    //! \param ref_points - a vector of reference points
    //! \return a vector of distances between consecutive reference points
    //!
    std::vector<T> CalculateDiffs(const std::vector<std::vector<T>> &ref_points) const;

    //! A method for calculation of t parameters and storing the data inside the spline
    //!
    //! \param diffs - a vector of distances between reference points
    //!
    void CalculateParams(const std::vector<T> diffs);

    //! A method for calculation of the spline knots and storing the data inside the spline
    void CalculateKnots();

    //! A method for calculation of one line of N matrix coefficients
    //!
    //! \param t - a value of the spline t parameter
    //! \return a vector of N matrix coefficients
    //!
    std::vector<T> CalculateCoeffs(T t) const;

    //! A method for calculation of N matrix coefficients (for all atored t parameters)
    //!
    //! \return a vector of full N matrix coefficients
    //!
    std::vector<T> CalculateCoeffsMatrix() const;

    //! A method for calculation of P and Q coefficients for a spline
    //!
    //! The method solves the spline, i. e. calculates P coefficients for spline points calculation
    //! and Q coefficients for spline derivative construction
    //! \param n - full spline N matrix
    //! \param ref_points - a vector of spline reference points
    //!
    void Solve(const std::vector<T> &n,
               const std::vector<std::vector<T>> &ref_points);

    //! A number of spline reference points
    size_t points_count_;
    //! A vector of spline t parameters
    std::vector<T> t_;
    //! A vector of spline knots
    std::vector<T> knots_;
    //! A vector of spline P coefficients
    std::vector<T> P_;
    //! A vector of spline Q coefficients
    std::vector<T> Q_;
};

//! The class for a B-spline with derivative
template <class T, int Degree>
class SplineD: public Spline<T, Degree> {
public:
    //! A default constructor
    SplineD() = default;

    //! A constructor on the basis of reference points
    //!
    //! \param ref_points - a vector of coordinates of referense points for a spline
    //!
    SplineD(const std::vector<std::vector<T>> &ref_points):
        Spline<T, Degree>(ref_points),
        derivative_(Spline<T, Degree - 1>(this->points_count_ - 1,
                                          {this->knots_.begin() + 1, this->knots_.end() - 1},
                                          this->Q_)) {}

    //! A method for returning the value of a derivative of a spline
    //!
    //! \param t - a parameter value
    //! \return a vector of derivative values in a point on a spline
    //!
    virtual std::vector<T> Derivative(T t) const override {
        auto result = derivative_(t);
        return result;
    }

    //! A method for returning the spline derivative as a Spline object
    //!
    //! \return a derivative spline object
    //!
    virtual const Spline<T, Degree - 1> &GetDerivativeSpline() const {
        return derivative_;
    }

protected:
    //! Derivative spline object
    const Spline<T, Degree - 1> derivative_;
};

} //! namespace spline
