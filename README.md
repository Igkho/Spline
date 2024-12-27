Spline
=========

Library for manipulating 2D parametric B-splines.

## Features

* Creation splines from a vector of points
* Creation of spiral-like splines
* Creation of ellipses
* Finding intersection points of two splines with an accuracy of 1e-10
* Finding the closest points of two splines with an accuracy 1e-10

## Building

Linux:

    $ git clone https://github.com/Igkho/Spline.git
    $ cd Spline
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make -j10

Windows:

* run CMake-gui;
* specify the source code and binary directories;
* press Configure button, set up the Generator settings (i.e. select the MSVC version);
* press Generate button;
* open ALL_BUILD.vcxproj in MSVC and build the solution;

Executables:

* Spline - the spline manipulation library
* SplineTests - unit tests of the library
* SplineExample - generates splines intersection and closest points finding examples


Dependencies:
* the library depends on Eigen (fetched by Cmake)
* tests depend on GoogleTest (fetched by Cmake)
* examples depend on OpenCV (to run the executable the OpenCV libraries should be on the search path)

Examples
--------

Intersecting two splines:

![multipoints](images/intersect_multipoint.gif "Intersecting two splines")

Intersecting a spline and an ellipse:

![ellipse](images/intersect_ellipse.gif "Intersecting spline and ellipses")

Intersecting spirals:

![spirals](images/intersect_spirals.gif "Intersecting spirals")

Closest points of two splines:

![closest](images/closest.gif "Closest points of two splines")

Closest points of a spline and an ellipse:

![clellipse](images/closest_ellipse.gif "Closest points of a spline and an ellipse")
