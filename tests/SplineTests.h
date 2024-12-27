#pragma once
#include <gtest/gtest.h>
#include "Spline.h"

using namespace spline;

TEST(SplineTest, ConstructorValidInput) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
    EXPECT_NO_THROW((Spline<double, 3>{ref_points}));
}

TEST(SplineTest, ConstructorInvalidInput) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {1, 1}, {2, 2, 2}};
    EXPECT_THROW((Spline<double, 3>{ref_points}), std::runtime_error);
    ref_points.resize(2);
    EXPECT_THROW((Spline<double, 3>{ref_points}), std::runtime_error);
}

TEST(SplineTest, TParameters) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
    Spline<double, 3> spline(ref_points);
    std::vector<double> N;
    auto params = spline.GetTParameters();
    EXPECT_EQ(params.size(), 5);
    EXPECT_NEAR(params[0], 0, 1e-14);
    EXPECT_NEAR(params[1], 0.25, 1e-14);
    EXPECT_NEAR(params[2], 0.5, 1e-14);
    EXPECT_NEAR(params[3], 0.75, 1e-14);
    EXPECT_NEAR(params[4], 1, 1e-14);
}

TEST(SplineTest, Knots) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
    Spline<double, 3> spline(ref_points);
    std::vector<double> N;
    auto knots = spline.GetKnots();
    EXPECT_EQ(knots.size(), 9);
    EXPECT_NEAR(knots[0], 0, 1e-14);
    EXPECT_NEAR(knots[1], 0, 1e-14);
    EXPECT_NEAR(knots[2], 0, 1e-14);
    EXPECT_NEAR(knots[3], 0, 1e-14);
    EXPECT_NEAR(knots[4], 0.5, 1e-14);
    EXPECT_NEAR(knots[5], 1, 1e-14);
    EXPECT_NEAR(knots[6], 1, 1e-14);
    EXPECT_NEAR(knots[7], 1, 1e-14);
    EXPECT_NEAR(knots[8], 1, 1e-14);
}

 TEST(SplineTest, PParameters) {
     std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
     Spline<double, 3> spline(ref_points);
     auto P = spline.GetPCoefficients();
    EXPECT_EQ(P.size(), 10);
    EXPECT_NEAR(P[0], 0, 1e-14);
    EXPECT_NEAR(P[1], 2, 1e-14);
    EXPECT_NEAR(P[2], 6, 1e-14);
    EXPECT_NEAR(P[3], 10, 1e-14);
    EXPECT_NEAR(P[4], 12, 1e-14);
    EXPECT_NEAR(P[5], 0, 1e-14);
    EXPECT_NEAR(P[6], 8./3, 1e-14);
    EXPECT_NEAR(P[7], 8, 1e-14);
    EXPECT_NEAR(P[8], 40./3, 1e-14);
    EXPECT_NEAR(P[9], 16, 1e-14);
}

 TEST(SplineTest, QParameters) {
     std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
     Spline<double, 3> spline(ref_points);
     auto Q = spline.GetQCoefficients();
     EXPECT_EQ(Q.size(), 8);
     EXPECT_NEAR(Q[0], 12, 1e-14);
     EXPECT_NEAR(Q[1], 12, 1e-14);
     EXPECT_NEAR(Q[2], 12, 1e-14);
     EXPECT_NEAR(Q[3], 12, 1e-14);
     EXPECT_NEAR(Q[4], 16, 1e-14);
     EXPECT_NEAR(Q[5], 16, 1e-14);
     EXPECT_NEAR(Q[6], 16, 1e-14);
     EXPECT_NEAR(Q[7], 16, 1e-14);
 }

TEST(SplineTest, CurvePoints) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}};
    Spline<double, 3> spline(ref_points);
    auto t = spline.GetTParameters();
    for (int i = 0; i < t.size(); ++i) {
        EXPECT_NEAR(spline(t[i])[0], ref_points[i][0], 1e-14);
        EXPECT_NEAR(spline(t[i])[1], ref_points[i][1], 1e-14);
    }
}

TEST(SplineTest, CurveDerivative) {
    std::vector<std::vector<double>> ref_points{{0, 0}, {3, 4}, {7, 7}, {10, 11}, {14, 14}};
    std::vector<std::vector<double>> ref_deriv{{6, 22}, {16, 12}, {14, 14}, {12, 16}, {22, 6}};
    SplineD<double, 3> spline(ref_points);
    auto t = spline.GetTParameters();
    for (int i = 0; i < t.size(); ++i) {
        auto d = spline.Derivative(t[i]);
        EXPECT_NEAR(d[0], ref_deriv[i][0], 1e-14);
        EXPECT_NEAR(d[1], ref_deriv[i][1], 1e-14);
    }
}

void CheckPointsIntersection(const BasicCurve<double> &spline1,
                             const BasicCurve<double> &spline2,
                             size_t ref_size,
                             const std::vector<std::vector<double>> &ref_intersections) {
    auto points = spline1.Intersect(spline2);
    EXPECT_EQ(points.size(), ref_size);
    for (size_t i = 0; i < points.size(); ++i) {
        EXPECT_NEAR(points[i][0], ref_intersections[i][0],
                    BasicCurve<double>::EPSILON * std::abs(points[i][0]));
        EXPECT_NEAR(points[i][1], ref_intersections[i][1],
                    BasicCurve<double>::EPSILON * std::abs(points[i][1]));
        auto cps = spline1(points[i][0]);
        auto cpe = spline2(points[i][1]);
        double dx = cps[0] - cpe[0];
        double dy = cps[1] - cpe[1];
        EXPECT_NEAR(std::sqrt(dx * dx + dy * dy), 0, BasicCurve<double>::EPSILON);
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

void CheckPointsClosest(const BasicCurve<double> &spline1,
                        const BasicCurve<double> &spline2,
                        const std::vector<std::vector<double>> & ref_close,
                        const std::vector<std::vector<double>> &ref_close_values) {
    auto points = spline1.Intersect(spline2);
    EXPECT_EQ(points.size(), 0);
    points = spline1.Closest(spline2);
    EXPECT_EQ(points.size(), 1);
    EXPECT_NEAR(points[0][0], ref_close[0][0],
                BasicCurve<double>::EPSILON * std::abs(points[0][0]));
    EXPECT_NEAR(points[0][1], ref_close[0][1],
                BasicCurve<double>::EPSILON * std::abs(points[0][1]));
    auto cp1 = spline1(points[0][0]);
    auto cp2 = spline2(points[0][1]);
    EXPECT_NEAR(cp1[0], ref_close_values[0][0],
                BasicCurve<double>::EPSILON * std::abs(cp1[0]));
    EXPECT_NEAR(cp1[1], ref_close_values[0][1],
                BasicCurve<double>::EPSILON * std::abs(cp1[1]));
    EXPECT_NEAR(cp2[0], ref_close_values[1][0],
                BasicCurve<double>::EPSILON * std::abs(cp2[0]));
    EXPECT_NEAR(cp2[1], ref_close_values[1][1],
                BasicCurve<double>::EPSILON * std::abs(cp2[1]));
}

TEST(SplineTest, ClosestTwoSplines) {
    std::vector<std::vector<double>> ref_points1{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    std::vector<std::vector<double>> ref_points2{{-8, -3}, {-4, 0}, {-1, 4}, {2, 8}, {6, 11}, {10, 14}};
    spline::SplineD<double, 3> spline1(ref_points1);
    spline::SplineD<double, 3> spline2(ref_points2);
    std::vector<std::vector<double>> ref_close0{{0.65403686511313253, 0.94596313488686712}};
    std::vector<std::vector<double>> ref_close1{{0.94596313488686712, 0.65403686511313253}};
    std::vector<std::vector<double>> ref_close_val0{{9.9935690192598834, 11.897721259699754},
                                                    {9.0064309807400917, 13.102278740300225}};
    std::vector<std::vector<double>> ref_close_val1{{9.0064309807400917, 13.102278740300225},
                                                    {9.9935690192598834, 11.897721259699754}};
    CheckPointsClosest(spline1, spline2, ref_close0, ref_close_val0);
    CheckPointsClosest(spline2, spline1, ref_close1, ref_close_val1);
}

TEST(SplineTest, ClosestTwoSplinesEnds) {
    std::vector<std::vector<double>> ref_points1{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    std::vector<std::vector<double>> ref_points2{{-2, 0}, {-6, 3}, {-9, 7}, {-12, 11}, {-16, 14}, {-20, 17}};
    spline::SplineD<double, 3> spline1(ref_points1);
    spline::SplineD<double, 3> spline2(ref_points2);
    std::vector<std::vector<double>> ref_close{{0, 0}};
    std::vector<std::vector<double>> ref_close_val0{{-1, 0}, {-2, 0}};
    std::vector<std::vector<double>> ref_close_val1{{-2, 0}, {-1, 0}};
    CheckPointsClosest(spline1, spline2, ref_close, ref_close_val0);
    CheckPointsClosest(spline2, spline1, ref_close, ref_close_val1);
}

TEST(SplineTest, ClosestSplineEllipse) {
    std::vector<std::vector<double>> ref_points{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    spline::SplineD<double, 3> spline(ref_points);
    spline::Ellipse<double> ellipse({20, 5}, {-15, 10});
    std::vector<std::vector<double>> ref_close0{{0.4438285236656746, 0.97372054218841175}};
    std::vector<std::vector<double>> ref_close1{{0.97372054218841175, 0.4438285236656746}};
    std::vector<std::vector<double>> ref_close_val0{{6.610886178837724, 7.9231121494608878},
                                                    {4.7279770223832038, 9.178152895746404}};
    std::vector<std::vector<double>> ref_close_val1{{4.7279770223832038, 9.178152895746404},
                                                    {6.610886178837724, 7.9231121494608878}};
    CheckPointsClosest(spline, ellipse, ref_close0, ref_close_val0);
    CheckPointsClosest(ellipse, spline, ref_close1, ref_close_val1);
}

TEST(SplineTest, ClosestSplineEllipseClose) {
    std::vector<std::vector<double>> ref_points{{-1, 0}, {3, 3}, {6, 7}, {9, 11}, {13, 14}, {17, 17}};
    spline::SplineD<double, 3> spline(ref_points);
    spline::Ellipse<double> ellipse({20, 5}, {-12.3, 10});
    std::vector<std::vector<double>> ref_close0{{0.50017022234140029, 0.9716406504596623}};
    std::vector<std::vector<double>> ref_close1{{0.9716406504596623, 0.50017022234140029}};
    std::vector<std::vector<double>> ref_close_val0{{7.4149944838495108, 9.0909632980994992},
                                                    {7.3833329545030324, 9.1137719042842686}};
    std::vector<std::vector<double>> ref_close_val1{{7.3833329545030324, 9.1137719042842686},
                                                    {7.4149944838495108, 9.0909632980994992}};
    CheckPointsClosest(spline, ellipse, ref_close0, ref_close_val0);
    CheckPointsClosest(ellipse, spline, ref_close1, ref_close_val1);
}
