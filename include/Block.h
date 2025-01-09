#pragma once
#include <vector>
#include <cuda_runtime_api.h>

namespace spline {

//! A class for storing consecutive elements of type T in the device memory
//! The functionality is close to std::vector
template <class T>
class Block {
public:
    //! The default constructor. Constructs an empty block
    Block() noexcept;

    //! Constructs the block of size elements length (memory is not initialized)
    Block(const size_t size);

    //! Constructs the block of size elements length and fills all bytes of memory with val
    Block(const size_t size, int val);

    //! The copy constructor. Constructs a block with the contents of other.
    Block(const Block<T> &other);

    //! Constructs a block from the host vector
    Block(const std::vector<T> &other);

    //! The move constructor. Constructs a block with the contents of other using move semantics.
    //! The data is moved from other into this container. other is empty afterwards
    Block(Block<T> &&other) noexcept;

    //! The default destructor
    ~Block();

    //! Copy assignment operator. Replaces the contents with a copy of the contents of other
    Block<T> &operator=(const Block<T> &other);

    //! Move assignment operator. Replaces the contents with those of other using move semantics
    //! (i.e. the data in other is moved from other into this container). other is empty afterwards
    Block<T> &operator=(Block<T> &&other);

    //! Replaces the data of the block with the host vector data
    void assign(const std::vector<T> &other);

    //! Returns the element at specified location pos
    __host__ __device__ const T &operator [](size_t pos) const;

    //! Returns a host vector constructed with the contents of the block
    std::vector<T> to_vector();

    //! Returns a pointer to the underlying array serving as element storage
    __host__ __device__ T *data() noexcept;

    //! Returns a const pointer to the underlying array serving as element storage
    __host__ __device__ const T *data() const noexcept;

    //! Returns a pointer to the first element of a block
    __host__ __device__ T *begin() noexcept;

    //! Returns a const pointer to the first element of a block
    __host__ __device__ const T *begin() const noexcept;

    //! Returns a const pointer to the first element of a block
    __host__ __device__ const T *cbegin() const noexcept;

    //! Returns a pointer to the element of a block following the last element
    __host__ __device__ T *end() noexcept;

    //! Returns a const pointer to the element of a block following the last element
    __host__ __device__ const T *end() const noexcept;

    //! Returns a const pointer to the element of a block following the last element
    __host__ __device__ const T *cend() const noexcept;

    //! Checks if the block has no elements. Returns true if the block is empty, false otherwise
    __host__ __device__ bool empty() const noexcept;

    //! Returns the number of elements in the block
    __host__ __device__ size_t size() const;

    //! Returns the size of memory used by the block elements in bytes
    __host__ __device__ size_t byte_size() const;

    //! Increase the capacity of the block (the total number of elements that the block can hold
    //! without requiring reallocation) to a value that's greater or equal to new_cap.
    //! If new_cap is greater than the current capacity(), new storage is allocated,
    //! otherwise the function does nothing. reserve() does not change the size of the block.
    void reserve(size_t new_cap);

    //! Returns the number of elements that the block has currently allocated space for
    __host__ __device__ size_t capacity() const;

    //! Erases all elements from the container. After this call, size() returns zero.
    //! Leaves the capacity() of the block unchanged
    void clear() noexcept;

    //! Resizes the block to contain count elements, does nothing if new_size == size().
    //! If the current size is greater than new_size, the block is reduced to its first new_size elements.
    //! If the current size is less than new_size, then additional not initialized elements are appended
    void resize(size_t new_size);

    //! Resizes the block to contain count elements, does nothing if new_size == size().
    //! If the current size is greater than new_size, the block is reduced to its first new_size elements.
    //! If the current size is less than new_size, then additional elements are appended.
    //! Every byte of memory for the appended elements is filled with val
    void resize(size_t new_size, int val);

    //! Exchanges the contents and capacity of the container with those of other
    void swap(Block<T> &other);

  private:
    //! A pointer to internal data storage
    T *ptr_;
    //! Size and capacity values
    size_t size_, capacity_;

    //! Frees the underlying memory. After this call, size() and capacity() return zero.
    void free();

    //! Allocates the memory for new_cap elements. Updates size and capacity to new_cap.
    void malloc(size_t new_cap);

    //! If possible (allocated memory is enough) copies the data from other to this block.
    //! No memory allocations are made.
    void copy_from(const Block<T> &other);

    //! If possible (allocated memory is enough) copies the data from other host vector to this block.
    //! No memory allocations are made.
    void copy_from(const std::vector<T> &other);

    //! If possible (the size of a host vector is enough) copies the data from this block to other host vector.
    //! No memory allocations are made.
    void copy_to(std::vector<T> &other);
};

} // namespace spline
