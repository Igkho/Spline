#pragma once
#include <gtest/gtest.h>
#include "Spline.h"
#include <chrono>
#include "../src/SplineSolver.h"
#include "../src/SearchInitializer.h"
#include <cmath>
#include "../src/ComplexFunctions.h"

using namespace spline;

constexpr double compare_precision = 1e-14; //1e-5;

TEST(SplineTest, ConstructorValidInput) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
    EXPECT_NO_THROW((SplineD<double, 3>{ref_points}));
}

TEST(SplineTest, ConstructorInvalidInput) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {1, 1}, {2, 2, 2}};
    EXPECT_THROW((SplineD<double, 3>{ref_points}), std::runtime_error);
    ref_points.resize(2);
    EXPECT_THROW((SplineD<double, 3>{ref_points}), std::runtime_error);
}

TEST(SplineTest, TParameters) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
    SplineSolver<double, 3> ss;
    auto [min_dparam, t] = ss.CalculateParams(ref_points);
    EXPECT_EQ(min_dparam, 0.25);
    EXPECT_EQ(t.size(), 5);
    EXPECT_NEAR(t[0], 0, compare_precision);
    EXPECT_NEAR(t[1], 0.25, compare_precision);
    EXPECT_NEAR(t[2], 0.5, compare_precision);
    EXPECT_NEAR(t[3], 0.75, compare_precision);
    EXPECT_NEAR(t[4], 1, compare_precision);
}

TEST(SplineTest, Knots) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
    SplineSolver<double, 3> ss;
    auto [min_dparam, t] = ss.CalculateParams(ref_points);
    auto knots = ss.CalculateKnots(t);
    EXPECT_EQ(knots.size(), 9);
    EXPECT_NEAR(knots[0], 0, compare_precision);
    EXPECT_NEAR(knots[1], 0, compare_precision);
    EXPECT_NEAR(knots[2], 0, compare_precision);
    EXPECT_NEAR(knots[3], 0, compare_precision);
    EXPECT_NEAR(knots[4], 0.5, compare_precision);
    EXPECT_NEAR(knots[5], 1, compare_precision);
    EXPECT_NEAR(knots[6], 1, compare_precision);
    EXPECT_NEAR(knots[7], 1, compare_precision);
    EXPECT_NEAR(knots[8], 1, compare_precision);
}

 TEST(SplineTest, PParameters) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
    SplineSolver<double, 3> ss;
    auto [min_dparam, knots, P, Q] = ss.SolveSpline(ref_points);
    EXPECT_EQ(P.size(), 10);
    EXPECT_NEAR(P[0], 0, compare_precision);
    EXPECT_NEAR(P[1], 2, compare_precision);
    EXPECT_NEAR(P[2], 6, compare_precision);
    EXPECT_NEAR(P[3], 10, compare_precision);
    EXPECT_NEAR(P[4], 12, compare_precision);
    EXPECT_NEAR(P[5], 0, compare_precision);
    EXPECT_NEAR(P[6], 8./3, compare_precision);
    EXPECT_NEAR(P[7], 8, compare_precision);
    EXPECT_NEAR(P[8], 40./3, compare_precision);
    EXPECT_NEAR(P[9], 16, compare_precision);
}

 TEST(SplineTest, QParameters) {
     std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
     SplineSolver<double, 3> ss;
     auto [min_dparam, knots, P, Q] = ss.SolveSpline(ref_points);
     EXPECT_EQ(Q.size(), 8);
     EXPECT_NEAR(Q[0], 12, compare_precision);
     EXPECT_NEAR(Q[1], 12, compare_precision);
     EXPECT_NEAR(Q[2], 12, compare_precision);
     EXPECT_NEAR(Q[3], 12, compare_precision);
     EXPECT_NEAR(Q[4], 16, compare_precision);
     EXPECT_NEAR(Q[5], 16, compare_precision);
     EXPECT_NEAR(Q[6], 16, compare_precision);
     EXPECT_NEAR(Q[7], 16, compare_precision);
 }

TEST(SplineTest, CurvePoints) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
    SplineSolver<double, 3> ss;
    auto [min_dparam, t] = ss.CalculateParams(ref_points);
    SplineD<double, 3> spline(ref_points);
    for (int i = 0; i < t.size(); ++i) {
        auto v = spline(t[i]);
        EXPECT_NEAR(v[0], ref_points[i][0], compare_precision);
        EXPECT_NEAR(v[1], ref_points[i][1], compare_precision);
    }
}

TEST(SplineTest, CurveDerivative) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {7, 7}, {10, 11}, {14, 14}};
    std::vector<std::vector<double>> ref_deriv{{6, 22}, {16, 12}, {14, 14}, {12, 16}, {22, 6}};
    SplineSolver<double, 3> ss;
    auto [min_dparam, t] = ss.CalculateParams(ref_points);
    SplineD<double, 3> spline(ref_points);
    for (int i = 0; i < t.size(); ++i) {
        auto d = spline.Derivative(t[i]);
        EXPECT_NEAR(d[0], ref_deriv[i][0], compare_precision);
        EXPECT_NEAR(d[1], ref_deriv[i][1], compare_precision);
    }
}

template <class Curve1, class Curve2>
void CheckSearchInitializer(Curve1 &c1,
                            Curve2 &c2,
                            double ratio,
                            const std::vector<double> &ref_vals) {
    spline::SearchInitializer<double, Curve1, Curve2> si(c1, c2);
    auto results = si.GetSearchPoints(ratio);
    EXPECT_EQ(results.size(), ref_vals.size());
    for (int i = 0; i < std::min(results.size(), ref_vals.size()); ++i) {
        L2Norm<double, Curve1, Curve2> l2n(c1, c2);
        EXPECT_NEAR(l2n(results[i][0], results[i][1])[0], ref_vals[i],
                   (Intersector<double, Curve1, Curve2>::EPSILON * std::abs(ref_vals[i])));
    }
}

TEST(SplineTest, SearchInitIntersect) {
    std::vector<std::vector<double>> ref_points1{{-1, 0}, {1, 3}, {-1, 7}, {1, 11}, {-1, 14}, {1, 17}, {-1, 20}};
    std::vector<std::vector<double>> ref_points2{{0, 0}, {-2, 3}, {0, 7}, {-2, 11}, {-0, 14}, {-2, 17}, {0, 20}};
    spline::SplineD<double, 3> spline1(ref_points1);
    spline::SplineD<double, 3> spline2(ref_points2);
    std::vector<double> ref_vals{0.064161112985266,
                                 0.23514960748908,
                                 0.54505941390908,
                                 0.78185731156511,
                                 0.84455969161699,
                                 1.0,
                                 1.0,
                                 1.0169565942594,
                                 2.2258101830676,
                                 2.2258101830676,
                                 2.3060777517305,
                                 2.3060777517305,
                                 2.6367586788225,
                                 2.6367586788225,
                                 3.0354531420999,
                                 3.0354531420999,
                                 3.2856695555019,
                                 3.2856695555019,
                                 3.5787168593497,
                                };
    CheckSearchInitializer(spline1, spline2,
                           Intersector<double,
                                       spline::SplineD<double, 3>,
                                       spline::SplineD<double, 3>>::INTERSECT_SEARCH_GRID_RATIO,
                           ref_vals);
    CheckSearchInitializer(spline2, spline1,
                           Intersector<double,
                                       spline::SplineD<double, 3>,
                                       spline::SplineD<double, 3>>::INTERSECT_SEARCH_GRID_RATIO,
                           ref_vals);
}

TEST(SplineTest, SearchInitClosest) {
    std::vector<std::vector<double>> ref_points{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    spline::SplineD<double, 3> spline(ref_points);
    spline::Ellipse<double> ellipse({20, 5}, {-12.3, 10});
    std::vector<double> ref_vals{0.016490084982884,
                                 0.016801328967701,
                                 0.018688252583091,
                                 0.018736965140641,
                                 0.024773205182818,
                                 0.050978045923057,
                                 0.056154030280996,
                                 0.067170003206751,
                                 0.10704445240542,
                                 0.11230672222222,
                                 0.11230672222222,
                                 0.12254368836806,
                                 0.12254368836806,
                                 0.13516460656584,
                                 0.13975536235957,
                                };
    CheckSearchInitializer(spline, ellipse,
                           Intersector<double, spline::SplineD<double, 3>,
                                               spline::Ellipse<double>>::CLOSEST_SEARCH_GRID_RATIO,
                           ref_vals);
    CheckSearchInitializer(ellipse, spline,
                           Intersector<double, spline::SplineD<double, 3>,
                                               spline::Ellipse<double>>::CLOSEST_SEARCH_GRID_RATIO,
                           ref_vals);
}

template <class Curve1, class Curve2>
void CheckPointsIntersection(Curve1 &sp1,
                             Curve2 &sp2,
                             size_t ref_size,
                             const std::vector<std::vector<double>> &ref_intersections) {
    std::vector<std::vector<std::vector<double>>> points{
        Intersector<double, Curve1, Curve2>(sp1, sp2).
            Intersect()
    };
    for (const auto &p : points) {
        EXPECT_EQ(p.size(), ref_size);
        for (size_t i = 0; i < p.size(); ++i) {
            EXPECT_NEAR(p[i][0], ref_intersections[i][0],
                        (Intersector<double, Curve1, Curve2>::EPSILON * std::abs(p[i][0])));
            EXPECT_NEAR(p[i][1], ref_intersections[i][1],
                        (Intersector<double, Curve1, Curve2>::EPSILON * std::abs(p[i][1])));
            auto cps = sp1(p[i][0]);
            auto cpe = sp2(p[i][1]);
            double dx = cps[0] - cpe[0];
            double dy = cps[1] - cpe[1];
            EXPECT_NEAR(std::sqrt(dx * dx + dy * dy), 0, (Intersector<double, Curve1, Curve2>::EPSILON));
        }
    }
}

TEST(SplineTest, IntersectTwoSplinesTouch) {
    std::vector<std::vector<double>> ref_points1{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    std::vector<std::vector<double>> ref_points2{{-1, 0}, {-5, 3}, {-8, 7}, {-11, 11}, {-15, 14}, {-19, 17}};
    spline::SplineD<double, 3> spline1(ref_points1);
    spline::SplineD<double, 3> spline2(ref_points2);
    std::vector<std::vector<double>> ref_ints{{0, 0}};
    CheckPointsIntersection(spline1, spline2, 1, ref_ints);
    CheckPointsIntersection(spline2, spline1, 1, ref_ints);
}

TEST(SplineTest, IntersectTwoSplinesMultipoint) {
    std::vector<std::vector<double>> ref_points1{{-1, 0}, {1, 3}, {-1, 7}, {1, 11}, {-1, 14}, {1, 17}, {-1, 20}};
    std::vector<std::vector<double>> ref_points2{{0, 0}, {-2, 3}, {0, 7}, {-2, 11}, {-0, 14}, {-2, 17}, {0, 20}};
    spline::SplineD<double, 3> spline1(ref_points1);
    spline::SplineD<double, 3> spline2(ref_points2);
    std::vector<std::vector<double>> ref_ints{{0.011790713774442185, 0.011790713774442185},
                                              {0.26220612568836521, 0.26220612568836521},
                                              {0.39471060285542309, 0.39471060285542309},
                                              {0.63957513236712926, 0.63957513236712926},
                                              {0.7562697904875586, 0.7562697904875586},
                                              {0.99006866038474262, 0.99006866038474262}};
    CheckPointsIntersection(spline1, spline2, 6, ref_ints);
    CheckPointsIntersection(spline2, spline1, 6, ref_ints);
}

TEST(SplineTest, IntersectSplineEllipse) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {4, 3}, {7, 7}, {10, 11}, {14, 14}, {18, 17}};
    spline::SplineD<double, 3> spline(ref_points);
    spline::Ellipse<double> ellipse({20, 5}, {0, 10});
    std::vector<std::vector<double>> ref_ints0{{0.31400479778872636, 0.79663675094864139},
                                               {0.77713794471440523, 0.13192151950636874}};
    std::vector<std::vector<double>> ref_ints1{{0.13192151950636874, 0.77713794471440523},
                                               {0.79663675094864139, 0.31400479778872636}};
    CheckPointsIntersection(spline, ellipse, 2, ref_ints0);
    CheckPointsIntersection(ellipse, spline, 2, ref_ints1);
}

template <class Curve1, class Curve2>
void CheckPointsClosest(Curve1 &sp1,
                        Curve2 &sp2,
                        const std::vector<std::vector<double>> &ref_close,
                        double ref_close_distance) {
    auto intor = Intersector<double, Curve1, Curve2>(sp1, sp2);
    auto ipoints = intor.Intersect();
    EXPECT_EQ(ipoints.size(), 0);
    std::vector<std::vector<std::vector<double>>> cpoints {
        intor.Closest(ClosestOptAlgorithm::RMSProp_cpu),
        intor.Closest(ClosestOptAlgorithm::BruteForce_cpu),
        intor.Closest(ClosestOptAlgorithm::BruteForce_cuda)
    };
    cuda::std::array<double, 2> max_val{0, 0};
    for (const auto &p : cpoints) {
        EXPECT_EQ(p.size(), 1);
        L2Norm<double, Curve1, Curve2> l2n(sp1, sp2);
        EXPECT_NEAR(p[0][0], ref_close[0][0],
                    (2 * Intersector<double, Curve1, Curve2>::EPSILON));
        EXPECT_NEAR(p[0][1], ref_close[0][1],
                    (2 * Intersector<double, Curve1, Curve2>::EPSILON));
        auto d = l2n(p[0][0], p[0][1])[0];
        auto dref = l2n(ref_close[0][0], ref_close[0][1])[0];
        EXPECT_NEAR(d, dref, (Intersector<double, Curve1, Curve2>::EPSILON * d));
        auto dd = std::abs(d - dref);
        if (dd != 0) {
            auto der = l2n.Derivative(p[0][0], p[0][1]);
            EXPECT_NEAR(p[0][0], ref_close[0][0],
                        (10 * dd / std::max(std::abs(der[0]), compare_precision * compare_precision)));
            EXPECT_NEAR(p[0][1], ref_close[0][1],
                        (10 * dd / std::max(std::abs(der[1]), compare_precision * compare_precision)));
        }
    }
}

template <class Curve1, class Curve2>
int CheckAlgorithmClosest(Curve1 &sp1,
                          Curve2 &sp2,
                          spline::ClosestOptAlgorithm alg) {
    auto intor = Intersector<double, Curve1, Curve2>(sp1, sp2);
    auto ipoints = intor.Intersect();
    EXPECT_EQ(ipoints.size(), 0);
    auto p0 = intor.Closest(ClosestOptAlgorithm::RMSProp_cpu, 8e-11);
    auto p1 = intor.Closest(alg);
    L2Norm<double, Curve1, Curve2> l2n(sp1, sp2);
    auto d0 = l2n(p0[0][0], p0[0][1])[0];
    auto d1 = l2n(p1[0][0], p1[0][1])[0];
    auto delta = std::abs(d0 - d1);
    auto dd = delta;
    if (dd != 0) {
        auto der = l2n.Derivative(p0[0][0], p0[0][1]);
        EXPECT_NEAR(p0[0][0], p1[0][0],
                    (dd / std::max(std::abs(der[0]), compare_precision * compare_precision)));
        EXPECT_NEAR(p0[0][1], p1[0][1],
                    (dd / std::max(std::abs(der[1]), compare_precision * compare_precision)));
    }
    return 0;
}

TEST(SplineTest, ClosestTwoSplines) {
    std::vector<std::vector<double>> ref_points1{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    std::vector<std::vector<double>> ref_points2{{-8, -3}, {-4, 0}, {-1, 4}, {2, 8}, {6, 11}, {10, 14}};
    spline::SplineD<double, 3> spline1(ref_points1);
    spline::SplineD<double, 3> spline2(ref_points2);
    std::vector<std::vector<double>> ref_close0{{0.65403686511313253, 0.94596313488686712}};
    std::vector<std::vector<double>> ref_close1{{0.94596313488686712, 0.65403686511313253}};
    double ref_close_distance = 2.4254002311633;
    CheckPointsClosest(spline1, spline2, ref_close0, ref_close_distance);
    CheckPointsClosest(spline2, spline1, ref_close1, ref_close_distance);
}

TEST(SplineTest, ClosestTwoSplinesEnds) {
    std::vector<std::vector<double>> ref_points1{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    std::vector<std::vector<double>> ref_points2{{-2, 0}, {-6, 3}, {-9, 7}, {-12, 11}, {-16, 14}, {-20, 17}};
    spline::SplineD<double, 3> spline1(ref_points1);
    spline::SplineD<double, 3> spline2(ref_points2);
    std::vector<std::vector<double>> ref_close{{0, 0}};
    double ref_close_distance = 1.0;
    CheckPointsClosest(spline1, spline2, ref_close, ref_close_distance);
    CheckPointsClosest(spline2, spline1, ref_close, ref_close_distance);
}

TEST(SplineTest, ClosestSplineEllipse) {
    std::vector<std::vector<double>> ref_points{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    spline::SplineD<double, 3> spline(ref_points);
    spline::Ellipse<double> ellipse({20, 5}, {-15, 10});
    std::vector<std::vector<double>> ref_close0{{0.4438285236656746, 0.97372054218841175}};
    std::vector<std::vector<double>> ref_close1{{0.97372054218841175, 0.4438285236656746}};
    double ref_close_distance = 5.1204741662972;
    CheckPointsClosest(spline, ellipse, ref_close0, ref_close_distance);
    CheckPointsClosest(ellipse, spline, ref_close1, ref_close_distance);
}

TEST(SplineTest, ClosestSplineEllipseClose) {
    std::vector<std::vector<double>> ref_points{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    spline::SplineD<double, 3> spline(ref_points);
    spline::Ellipse<double> ellipse({20, 5}, {-12.3, 10});
    std::vector<std::vector<double>> ref_close0{{0.50017022234140029, 0.9716406504596623}};
    std::vector<std::vector<double>> ref_close1{{0.9716406504596623, 0.50017022234140029}};
    double ref_close_distance = 0.0015226849566498;
    CheckPointsClosest(spline, ellipse, ref_close0, ref_close_distance);
    CheckPointsClosest(ellipse, spline, ref_close1, ref_close_distance);
}

TEST(SplineTest, ClosestSplineEllipseMove) {
    int errors = 0;
    for (size_t i = 0; i < 40; ++i) {
        const double shift = i * 0.1;
        std::vector<std::vector<double>> ref_points{{-1 + shift, 0}, {3 + shift, 3}, {6 + shift, 7}, {9 + shift, 11},
                                                    {13 + shift, 14}, {17 + shift, 17}};
        spline::SplineD<double, 3> sp(ref_points);
        spline::Ellipse<double> el({10, 2.5}, {-7.5 - shift, 5});
        errors += CheckAlgorithmClosest(el, sp, ClosestOptAlgorithm::BruteForce_cpu);
        errors += CheckAlgorithmClosest(sp, el, ClosestOptAlgorithm::BruteForce_cpu);
        errors += CheckAlgorithmClosest(el, sp, ClosestOptAlgorithm::BruteForce_cuda);
        errors += CheckAlgorithmClosest(sp, el, ClosestOptAlgorithm::BruteForce_cuda);
    }
}

constexpr size_t MEASURE_REPEAT_COUNT = 10;

template <class Curve1, class Curve2>
auto GetClosestTime(Intersector<double, Curve1, Curve2> &intor,
                    ClosestOptAlgorithm alg,
                    size_t count = MEASURE_REPEAT_COUNT) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
        intor.Closest(alg);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / count;
}

TEST(SplineTest, ClosestSpeed) {
    std::vector<std::vector<double>> ref_points{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    spline::SplineD<double, 3> spline(ref_points);
    spline::Ellipse<double> ellipse({20, 5}, {-12.3, 10});
    Intersector<double, spline::SplineD<double, 3>, spline::Ellipse<double>> intor(spline, ellipse);
    GetClosestTime(intor, ClosestOptAlgorithm::BruteForce_cuda);
    auto rp_cpu_time = GetClosestTime(intor, ClosestOptAlgorithm::RMSProp_cpu);
    auto bf_cpu_time = GetClosestTime(intor, ClosestOptAlgorithm::BruteForce_cpu);
    auto bf_cuda_time = GetClosestTime(intor, ClosestOptAlgorithm::BruteForce_cuda);
    std::cout << "Closest points calculation times:\nRMSProp CPU - "
              << rp_cpu_time << " us\nBrute force CPU - " << bf_cpu_time
              << " us\nBrute force CUDA - " << bf_cuda_time << " us" << std::endl;
    EXPECT_LT(bf_cuda_time, bf_cpu_time);
    EXPECT_LT(bf_cuda_time, rp_cpu_time);
}
