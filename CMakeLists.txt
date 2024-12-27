cmake_minimum_required(VERSION 3.19...3.31)

project(Spline
        VERSION 1.0
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)
FetchContent_MakeAvailable(Eigen)

find_package(OpenMP)

find_package(OpenCV REQUIRED)

find_package(Doxygen
             REQUIRED dot
             OPTIONAL_COMPONENTS mscgen dia)

file(GLOB_RECURSE DOC_SOURCES
     include/*.h
)

file(GLOB_RECURSE LIB_SOURCES
     src/*.cpp
     src/*.h
     include/*.h
)

file(GLOB_RECURSE EXAMPLE_SOURCES
     example/*.cpp
     include/*.h
)

file(GLOB_RECURSE TESTS
     include/*.h
     tests/*.cpp
     tests/*.h
)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(TESTS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tests)

add_library(Spline STATIC ${LIB_SOURCES})
set_target_properties(Spline PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(Spline PROPERTIES PUBLIC_HEADER ${INCLUDE_DIR}/Spline.h)

add_executable(SplineExample ${EXAMPLE_SOURCES})

add_executable(SplineTests ${TESTS})

set(DOXYGEN_OUTPUT_DIRECTORY docs)

doxygen_add_docs(SplineDocs
    ${DOC_SOURCES}
    ALL
)

target_include_directories(Spline PRIVATE ${INCLUDE_DIR})
target_include_directories(SplineExample PRIVATE ${INCLUDE_DIR})
target_include_directories(SplineTests PRIVATE ${TESTS_INCLUDE_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(Spline PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(Spline PRIVATE Eigen3::Eigen)

target_include_directories(SplineExample PRIVATE "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(SplineExample PRIVATE ${OpenCV_LIBS} )
target_link_libraries(SplineExample PRIVATE Spline )

target_link_libraries(SplineTests PRIVATE GTest::gtest_main)
target_link_libraries(SplineTests PRIVATE Spline)
