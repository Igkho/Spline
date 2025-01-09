#pragma once
#include "Intersector.h"
#include <vector>

namespace spline {

// A class for curves search initializer
template <class T, class Curve0, class Curve1>
class SearchInitializer {
public:
    static constexpr T MAX_DPARAM = (T)0.5;
    static constexpr T MIN_DPARAM = (T)0.01;

    SearchInitializer(const Curve0 &curve0,
                      const Curve1 &curve1) : curve0_(curve0), curve1_(curve1) {}

    std::vector<std::vector<T>> GetSearchPoints(T search_ratio = 1,
                        T select_ratio = Intersector<T, Curve0, Curve1>::SELECT_RATIO
                       ) const;

protected:
    const Curve0 &curve0_;
    const Curve1 &curve1_;
    std::vector<std::vector<T>> GetSearchGrid(const T ratio = 1) const;
};

} // namespace spline
