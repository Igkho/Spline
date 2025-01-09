#include <iostream>
#include <vector>
#include <array>
#include "Spline.h"
#include "opencv2/opencv.hpp"
#include <optional>
#define _USE_MATH_DEFINES
#include <math.h>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;

constexpr size_t TOTAL_DRAW_POINTS_COUNT = 50;
const cv::Scalar DEFAULT_CURVE_COLOR = cv::Scalar(0xBF, 0x43, 0x1A);

template <class Curve>
int DrawCurve(cv::Mat &img,
              spline::IBaseCurve<double, Curve> &curve,
              int scale,
              int shiftx,
              int shifty,
              const cv::Scalar &color = DEFAULT_CURVE_COLOR,
              int total_count = TOTAL_DRAW_POINTS_COUNT) {
    std::optional<std::vector<double>> last_point;
    for (size_t i = 0; i <= total_count; ++i) {
        double t = (double)i / total_count;
        auto p = curve(t);
        p[0] *= scale;
        p[1] *= scale;
        p[0] += shiftx;
        p[1] += shifty;
        if (last_point.has_value()) {
            cv::line(img, cv::Point(last_point.value()[0], last_point.value()[1]),
                     cv::Point(p[0], p[1]),
                     color);
        }
        last_point = {p[0], p[1]};
    }
    return 0;
}

template <class Curve0, class Curve1>
void DrawPoints(cv::Mat &img,
                const std::vector<std::vector<double>> &points,
                spline::IBaseCurve<double, Curve0> &curve0,
                spline::IBaseCurve<double, Curve1> &curve1,
                int scale, int shiftx, int shifty, double csize = 3) {
    for (const auto &p : points) {
        auto cp0 = curve0(p[0]);
        auto cp1 = curve1(p[1]);
        auto red_color = cv::Scalar(0x4B, 0x4F, 0xFF);
        spline::Ellipse<double> ellipse0({csize, csize}, {cp0[0], cp0[1]});
        DrawCurve(img, ellipse0, scale, shiftx, shifty, red_color);
        spline::Ellipse<double> ellipse1({csize, csize}, {cp1[0], cp1[1]});
        DrawCurve(img, ellipse1, scale, shiftx, shifty, red_color);
    }
}

void WriteImages(const std::vector<cv::Mat> &images,
                 const std::string &name) {
    for (size_t i = 0; i < images.size(); ++i) {
        std::stringstream ss, ss_mirr;
        ss << std::setw(3) << std::setfill('0') << i;
        ss_mirr << std::setw(3) << std::setfill('0') << images.size() * 2 - i;
        cv::imwrite(name + ss.str() + ".jpg", images[i]);
        cv::imwrite(name + ss_mirr.str() + ".jpg", images[i]);
    }
}

void IntersectSpirals(std::vector<cv::Mat> &images) {
    for (size_t i = 0; i < images.size(); ++i) {
        const double shift = i * 4;
        std::vector<std::vector<double>> ref_points1, ref_points2;
        constexpr size_t number_of_circles = 3;
        constexpr size_t points_per_circle = 8;
        size_t total_count = number_of_circles * points_per_circle;
        constexpr size_t max_radius = 200;
        constexpr size_t min_radius = 100;
        for (size_t i = 0; i <= total_count; ++i) {
            double arg = i * 2 * M_PI / points_per_circle;
            double radius = min_radius + (double)i * (max_radius - min_radius) / total_count;
            ref_points1.push_back({std::cos(arg) * radius - 50 - shift, std::sin(arg) * radius});
            ref_points2.push_back({std::cos(arg) * radius + 50 + shift, std::sin(arg) * radius});
        }
        spline::SplineD<double, 3> sp1(ref_points1);
        spline::SplineD<double, 3> sp2(ref_points2);
        DrawCurve(images[i], sp1, 1, 450, 260, DEFAULT_CURVE_COLOR, 1000);
        DrawCurve(images[i], sp2, 1, 450, 260, DEFAULT_CURVE_COLOR, 1000);
        std::vector<std::vector<double>> points =
            spline::Intersector<double, decltype(sp1), decltype(sp2)>(sp1, sp2).Intersect();
        DrawPoints(images[i], points, sp1, sp2, 1, 450, 260);
    }
}

void IntersectTwoSplinesTouch(std::vector<cv::Mat> &images) {
    for (size_t i = 0; i < images.size(); ++i) {
        const double shift = i * 0.1;
        std::vector<std::vector<double>> ref_points1{{-1 + shift, 0}, {3 + shift, 3}, {6 + shift, 7},
                                                     {9 + shift, 11}, {13 + shift, 14}, {17 + shift, 17}};
        std::vector<std::vector<double>> ref_points2{{-1 - shift, 0}, {-5 - shift, 3}, {-8 - shift, 7},
                                                     {-11 - shift, 11}, {-15 - shift, 14}, {-19 - shift, 17}};
        spline::SplineD<double, 3> sp1(ref_points1);
        spline::SplineD<double, 3> sp2(ref_points2);
        DrawCurve(images[i], sp1, 20, 450, 100);
        DrawCurve(images[i], sp2, 20, 450, 100);
        std::vector<std::vector<double>> points =
            spline::Intersector<double, decltype(sp1), decltype(sp2)>(sp1, sp2).Intersect();
        DrawPoints(images[i], points, sp1, sp2, 20, 450, 100, 0.2);
    }
}

void IntersectTwoSplinesMultipoint(std::vector<cv::Mat> &images) {
    for (size_t i = 0; i < images.size(); ++i) {
        const double shift = (int)(i - images.size() / 2) * 0.1;
        std::vector<std::vector<double>> ref_points1{{-1 + shift, 0}, {1 + shift, 3}, {-1 + shift, 7},
                                                     {1 + shift, 11}, {-1 + shift, 14}, {1 + shift, 17}, {-1 + shift, 20}};
        std::vector<std::vector<double>> ref_points2{{0 - shift, 0}, {-2 - shift, 3}, {0 - shift, 7},
                                                     {-2 - shift, 11}, {-0 - shift, 14}, {-2 - shift, 17}, {0 - shift, 20}};
        spline::SplineD<double, 3> sp1(ref_points1);
        spline::SplineD<double, 3> sp2(ref_points2);
        DrawCurve(images[i], sp1, 20, 450, 50);
        DrawCurve(images[i], sp2, 20, 450, 50);
        std::vector<std::vector<double>> points =
            spline::Intersector<double, decltype(sp1), decltype(sp2)>(sp1, sp2).Intersect();
        DrawPoints(images[i], points, sp1, sp2, 20, 450, 50, 0.2);
    }
}

void IntersectSplineEllipse(std::vector<cv::Mat> &images) {
    for (size_t i = 0; i < images.size(); ++i) {
        const double shift = i * 0.1;
        std::vector<std::vector<double>> ref_points{{0 + shift, 0}, {4 + shift, 3}, {7 + shift, 7}, {10 + shift, 11},
                                                    {14 + shift, 14}, {18 + shift, 17}};
        spline::SplineD<double, 3> sp(ref_points);
        spline::Ellipse<double> el({12, 2.5}, {0 - shift, 5});
        DrawCurve(images[i], sp, 20, 450, 50);
        DrawCurve(images[i], el, 20, 450, 50);
        std::vector<std::vector<double>> points =
            spline::Intersector<double, decltype(el), decltype(sp)>(el, sp).Intersect();
        DrawPoints(images[i], points, el, sp, 20, 450, 50, 0.2);
    }
}

void ClosestTwoSplines(std::vector<cv::Mat> &images) {
    for (size_t i = 0; i < images.size(); ++i) {
        const double shift = (int)(i - images.size() / 4) * 0.1;
        std::vector<std::vector<double>> ref_points1{{-1 + shift, 3}, {3 + shift, 6}, {6 + shift, 10}, {9 + shift, 14},
                                                     {13 + shift, 17}, {17 + shift, 20}};
        std::vector<std::vector<double>> ref_points2{{-8 - shift, 3}, {-4 - shift, 6}, {-1 - shift, 10}, {2 - shift, 14},
                                                     {6 - shift, 17}, {10 - shift, 20}};
        spline::SplineD<double, 3> sp1(ref_points1);
        spline::SplineD<double, 3> sp2(ref_points2);
        DrawCurve(images[i], sp1, 20, 450, 50);
        DrawCurve(images[i], sp2, 20, 450, 50);
        std::vector<std::vector<double>> points =
            spline::Intersector<double, decltype(sp1), decltype(sp2)>(sp1, sp2).Closest();
        DrawPoints(images[i], points, sp1, sp2, 20, 450, 50, 0.2);
    }
}

void ClosestTwoSplinesEnds(std::vector<cv::Mat> &images) {
    for (size_t i = 0; i < images.size(); ++i) {
        const double shift = i * 0.1;
        std::vector<std::vector<double>> ref_points1{{-1 + shift, 0}, {3 + shift, 3}, {6 + shift, 7}, {9 + shift, 11},
                                                     {13 + shift, 14}, {17 + shift, 17}};
        std::vector<std::vector<double>> ref_points2{{-2 - shift, 0}, {-6 - shift, 3}, {-9 - shift, 7}, {-12 - shift, 11},
                                                     {-16 - shift, 14}, {-20 - shift, 17}};
        spline::SplineD<double, 3> sp1(ref_points1);
        spline::SplineD<double, 3> sp2(ref_points2);
        DrawCurve(images[i], sp1, 20, 450, 50);
        DrawCurve(images[i], sp2, 20, 450, 50);
        std::vector<std::vector<double>> points =
            spline::Intersector<double, decltype(sp1), decltype(sp2)>(sp1, sp2).Closest();
        DrawPoints(images[i], points, sp1, sp2, 20, 450, 50, 0.2);
    }
}

void ClosestSplineEllipse(std::vector<cv::Mat> &images) {
    for (size_t i = 0; i < images.size(); ++i) {
        const double shift = i * 0.1;
        std::vector<std::vector<double>> ref_points{{-1 + shift, 0}, {3 + shift, 3}, {6 + shift, 7}, {9 + shift, 11},
                                                    {13 + shift, 14}, {17 + shift, 17}};
        spline::SplineD<double, 3> sp(ref_points);
        spline::Ellipse<double> el({10, 2.5}, {-7.5 - shift, 5});
        DrawCurve(images[i], sp, 20, 450, 50);
        DrawCurve(images[i], el, 20, 450, 50);
        std::vector<std::vector<double>> points =
            spline::Intersector<double, decltype(el), decltype(sp)>(el, sp).Closest();
        DrawPoints(images[i], points, el, sp, 20, 450, 50, 0.2);
    }
}

void ClearImages(std::vector<cv::Mat> &images) {
    for (int i = 0; i < images.size(); ++i) {
        images[i] = cv::Scalar(255, 255, 255);
    }
}

int main() {
    constexpr int NUMBER_OF_IMAGES = 40;
    std::vector<cv::Mat> images(NUMBER_OF_IMAGES);
    for (int i = 0; i < images.size(); ++i) {
        images[i] = cv::Mat(540, 910, CV_8UC3);
    }

    std::vector<std::pair<std::string, std::function<void(std::vector<cv::Mat> &)>>> names{
        {"intersect_multipoint", IntersectTwoSplinesMultipoint},
        {"intersect_ellipse", IntersectSplineEllipse},
        {"intersect_spirals", IntersectSpirals},
        {"closest", ClosestTwoSplines},
        {"closest_ellipse", ClosestSplineEllipse}};

    for (const auto &name : names) {
        auto p = fs::path(name.first);
        if (!fs::exists(p)) {
            fs::create_directory(p);
        }
        ClearImages(images);
        name.second(images);
        WriteImages(images, (p/name.first).string());
    }
  return 0;
}


