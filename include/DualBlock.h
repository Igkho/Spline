#pragma once
#include <stdint.h>
#include <vector>
#include "Block.h"
#include <cuda_runtime_api.h>

namespace spline {

//! A class for storing consecutive elements of type T in both host and device memory
//! The functionality is close to std::vector
template <class T>
class DualBlock: public std::vector<T> {
public:
    //! The default constructor. Constructs an empty dual block
    DualBlock() : block_cpu_{}, block_cuda_{}, sync_(true) {}

    //! The default constructor. Constructs a dual block from the host vector
    DualBlock(const std::vector<T> &v) : block_cpu_(v), block_cuda_{}, sync_(false) {}

    //! A default destructor
    ~DualBlock() = default;

    //! Returns the element at specified location pos
    __host__ __device__ const T &operator[](size_t pos) const;

    //! Returns a const pointer to the underlying array serving as element storage
    __host__ __device__ const T *data() const noexcept;

    //! Returns a const pointer to the first element of a block
    __host__ __device__ const T *begin() const noexcept;

    //! Returns a const pointer to the first element of a block
    __host__ __device__ const T *cbegin() const noexcept;

    //! Returns a const pointer to the element of a block following the last element
    __host__ __device__ const T *end() const noexcept;

    //! Returns a const pointer to the element of a block following the last element
    __host__ __device__ const T *cend() const noexcept;

    //! Returns the number of elements in the block
    __host__ __device__ size_t size() const;

    //! Returns the size of memory used by either host or device block elements in bytes
    __host__ __device__ size_t byte_size() const;

    //! Returns the constant reference to the device block
    __host__ __device__ const Block<T> &GetBlockCuda() const {
        return block_cuda_;
    }

    //! If the host-device synchronization is false the data from host vector is copied to the device block
    void sync();

    //! Sets the host-device synchronization to false
    void reset_sync() {
        sync_ = false;
    }

  private:
    //! A host-device synchronization value
    bool sync_;
    //! A host data storage vector
    std::vector<T> block_cpu_;
    //! A device data storage block
    Block<T> block_cuda_;
};

} // namespace spline
