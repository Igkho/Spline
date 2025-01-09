#pragma once
#include "Block.h"
#include <cuda_runtime_api.h>

namespace spline {

//! Enum class for closest points algorithm
enum class ClosestOptAlgorithm {RMSProp_cpu, BruteForce_cpu, BruteForce_cuda};

//! A class for finding intersections or closest points of two curves
template <class T, class Curve0, class Curve1>
class Intersector {
public:
    //! A default relative precision of points calculation
    static constexpr T EPSILON = 1e-9;
    //! A default maximum number of iterations while finding curves intersections
    static constexpr size_t MAX_INTERSECTION_ITERS = 20;
    //! A default maximum number of RMSProp optimizer iterations while finding curves closest points
    static constexpr size_t MAX_CLOSEST_ITERS = 4000;
    //! A default alpha value for RMSProp optimizer (finding closest points)
    static constexpr T CLOSEST_RMSPROP_ALPHA = 0.0001;
    //! A default beta value for RMSProp optimizer (finding closest points)
    static constexpr T CLOSEST_RMSPROP_BETA = 0.9999;
    //! A default ratio for selecting initial points (finding intersection points)
    static constexpr size_t SELECT_RATIO = 3;
    //! A default ratio for init search grid calculation (finding curves intersections)
    static constexpr T INTERSECT_SEARCH_GRID_RATIO = 0.5;
    //! A default ratio for init search grid calculation (finding closest points)
    static constexpr T CLOSEST_SEARCH_GRID_RATIO = 0.05;
    //! A default power split value for brute force algorithm (finding closest points)
    static constexpr T CLOSEST_BRUTE_FORCE_POWER_SPLIT = 11;
    //! A default power step for brute force algorithm (finding closest points)
    static constexpr T CLOSEST_BRUTE_FORCE_POWER_STEP = 1.51;

    //! A constructor from two curves
    Intersector(Curve0 &curve0, Curve1 &curve1) :
        curve0_(curve0), curve1_(curve1) {}

    //! A method to find intersections of two curves on the basis of the Newton-Raphson method
    //!
    //! \param epsilon - relative precision of points calculation
    //! \param max_iters - maximum number of iterations while finding curves intersections for each point on a search grid
    //! \return a vector of points (each point is a vector of coordinates) of found curves intersections
    //!
    std::vector<std::vector<T>> Intersect(T epsilon = EPSILON,
                                          size_t max_iters = MAX_INTERSECTION_ITERS
                                         );

    //! The method to find closest points of two curves on the basis of the RMSProp optimization
    //!
    //! \param alg - the optimization algorith to use for calculations
    //! \param epsilon - relative precision of points calculation
    //! \param max_iters - maximum number of iterations while finding curves closest points for each point on a search grid
    //! \param alpha - alpha value for RMSProp optimizer
    //! \param beta - beta value for RMSProp optimizer
    //! \param p_split - power split value for BruteForce algorithm
    //! \param p_step - power step value for BruteForce algorithm
    //! \return a vector of two points (each point is a vector of coordinates) which are found to be the closest
    //!
    std::vector<std::vector<T>> Closest(ClosestOptAlgorithm alg = ClosestOptAlgorithm::BruteForce_cuda,
                                        T epsilon = EPSILON,
                                        size_t max_iters = MAX_CLOSEST_ITERS,
                                        T alpha = CLOSEST_RMSPROP_ALPHA,
                                        T beta = CLOSEST_RMSPROP_BETA,
                                        size_t p_split = CLOSEST_BRUTE_FORCE_POWER_SPLIT,
                                        T p_step = CLOSEST_BRUTE_FORCE_POWER_STEP);

private:
    //! The first curve
    Curve0 &curve0_;
    //! The second curve
    Curve1 &curve1_;
    //! A cuda buffer for optimization results data
    Block<T> results_cuda_;
};

} //! namespace spline
