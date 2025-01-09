#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <cuda_runtime_api.h>

#define ERROR_SOURCE spline::ErrorSource(__FILE__, __LINE__)

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cerr \
        << "Cuda error in " << ERROR_SOURCE << ": " \
        << cudaGetErrorString(err) << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cerr \
        << "Cudnn error in " << ERROR_SOURCE << ": " \
        << cudnnGetErrorString(err) << std::endl; \
    std::exit(1); \
  } \
}

#define CUDA_GENERAL_ERROR(f) { \
  std::cerr << "General error - " << (f) << ": " << ERROR_SOURCE << std::endl; \
  std::exit(1); \
}

namespace spline {

static inline std::string ErrorSource(const char *file, int line) {
  std::stringstream ss;
  ss << file << ":" << line;
  return ss.str();
}

class KernelGrid {
  public:
    KernelGrid() = delete;
    KernelGrid(const KernelGrid &) = default;
    KernelGrid &operator =(const KernelGrid &) = default;
    ~KernelGrid() = default;

    KernelGrid(unsigned int size, unsigned int block = 256) {
        calculate(dim3{size, 1, 1}, dim3{block, 1, 1});
    }

    KernelGrid(dim3 size, dim3 block) {
        calculate(size, block);
    }

    dim3 gsize() const {
        return gsize_;
    }

    dim3 bsize() const {
        return bsize_;
    }

  private:
    void calculate(const dim3 &size,
                   const dim3 &block) {
        bsize_ = dim3{std::max(1u, block.x), std::max(1u, block.y), std::max(1u, block.z)};
        gsize_ = dim3{(std::max(1u, size.x) + bsize_.x - 1) / bsize_.x,
                      (std::max(1u, size.y) + bsize_.y - 1) / bsize_.y,
                      (std::max(1u, size.z) + bsize_.z - 1) / bsize_.z};
    }

    dim3 gsize_, bsize_;
};

} // namespace spline
