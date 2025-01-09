#include "Block.h"
#include "helpers.h"
#include "SplineD.h"
#include "Ellipse.h"
#include "CurveCuda.h"

namespace spline {

template <class T>
Block<T>::Block() noexcept : ptr_(nullptr), size_(0), capacity_(0) {}

template <class T>
Block<T>::Block(const size_t size) : Block() {
    malloc(size);
}

template <class T>
Block<T>::Block(const size_t size, int val) : Block() {
    malloc(size);
    CUDA_CALL(cudaMemset(ptr_, val, byte_size()));
}

template <class T>
Block<T>::Block(const Block<T> &other) : Block(other.size()) {
    copy_from(other);
}

template <class T>
Block<T>::Block(const std::vector<T> &other) : Block(other.size()) {
    copy_from(other);
}

template <class T>
Block<T>::Block(Block<T> &&other) noexcept : Block() {
    ptr_ = other.ptr_;
    capacity_ = other.capacity_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.capacity_ = other.size_ = 0;
}

template <class T>
Block<T> &Block<T>::operator=(const Block<T> &other){
    resize(other.size());
    copy_from(other);
    return *this;
}

template <class T>
Block<T> &Block<T>::operator=(Block<T> &&other){
    free();
    ptr_ = other.ptr_;
    capacity_ = other.capacity_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.capacity_ = other.size_ = 0;
    return *this;
}

template <class T>
const T &Block<T>::operator[](size_t pos) const {
    return ptr_[pos];
}

template <class T>
Block<T>::~Block<T>() {
    free();
}

template <class T>
void Block<T>::assign(const std::vector<T> &other) {
    resize(other.size());
    copy_from(other);
}

template <class T>
std::vector<T> Block<T>::to_vector() {
    std::vector<T> result(this->size_);
    copy_to(result);
    return result;
}


template <class T>
T *Block<T>::data() noexcept {
    return ptr_;
}

template <class T>
const T *Block<T>::data() const noexcept {
    return ptr_;
}

template <class T>
T *Block<T>::begin() noexcept {
    return ptr_;
}

template <class T>
const T *Block<T>::begin() const noexcept {
    return ptr_;
}

template <class T>
const T *Block<T>::cbegin() const noexcept {
    return ptr_;
}

template <class T>
T *Block<T>::end() noexcept {
    return ptr_ + size_;
}

template <class T>
const T *Block<T>::end() const noexcept {
    return ptr_ + size_;
}

template <class T>
const T *Block<T>::cend() const noexcept {
    return ptr_ + size_;
}

template <class T>
bool Block<T>::empty() const noexcept {
    return !size_;
}

template <class T>
size_t Block<T>::size() const {
    return size_;
}

template <class T>
size_t Block<T>::byte_size() const {
    return size_ * sizeof(T);
}

template <class T>
void Block<T>::reserve(size_t new_cap) {
    if (capacity_ >= new_cap) {
        return;
    }
    Block<T> other(new_cap);
    other.copy_from(*this);
    swap(other);
}

template <class T>
size_t Block<T>::capacity() const {
    return capacity_;
}

template <class T>
void Block<T>::clear() noexcept {
    size_ = 0;
}

template <class T>
void Block<T>::resize(size_t new_size) {
    if (capacity_ >= new_size) {
        size_ = new_size;
        return;
    }
    Block<T> other(new_size);
    other.copy_from(*this);
    swap(other);
}

template <class T>
void Block<T>::resize(size_t new_size, int val) {
    if (capacity_ >= new_size) {
        if (new_size > size_) {
            CUDA_CALL(cudaMemset(ptr_ + size_, val, (new_size - size_) * sizeof(T)));
        }
        size_ = new_size;
        return;
    }
    Block<T> other(new_size);
    other.copy_from(*this);
    CUDA_CALL(cudaMemset(other.ptr_ + size_, val, (new_size - size_) * sizeof(T)));
    swap(other);
}

template <class T>
void Block<T>::swap(Block<T> &other) {
    std::swap(ptr_, other.ptr_);
    std::swap(capacity_, other.capacity_);
    std::swap(size_, other.size_);
}

template <class T>
void Block<T>::free() {
    if (ptr_ != nullptr) {
        CUDA_CALL(cudaFree((void *)ptr_));
        ptr_ = nullptr;
    }
    capacity_ = size_ = 0;
}

template <class T>
void Block<T>::malloc(size_t new_cap) {
    if (capacity_ >= new_cap) {
        size_ = new_cap;
        return;
    }
    free();
    CUDA_CALL(cudaMalloc((void **)&ptr_, new_cap * sizeof(T)));
    capacity_ = size_ = new_cap;
}

template <class T>
void Block<T>::copy_from(const Block<T> &other) {
    if (other.size() && capacity_ >= other.size()) {
        size_ = other.size();
        CUDA_CALL(cudaMemcpy(ptr_, other.data(), byte_size(), cudaMemcpyDeviceToDevice));
    }
}

template <class T>
void Block<T>::copy_from(const std::vector<T> &other) {
    if (other.size() && capacity_ >= other.size()) {
        size_ = other.size();
        CUDA_CALL(cudaMemcpy(ptr_, other.data(), byte_size(), cudaMemcpyHostToDevice));
    }
}

template <class T>
void Block<T>::copy_to(std::vector<T> &other) {
    if (size_ && other.size() >= size_) {
        CUDA_CALL(cudaMemcpy(other.data(), ptr_, byte_size(), cudaMemcpyDeviceToHost));
    }
}

template class Block<double>;
template class Block<SplineD<double, 3>>;
template class Block<Ellipse<double>>;
template class Block<CurveCuda<double>>;

template class Block<float>;
template class Block<CurveCuda<float>>;

template class Block<int>;
template class Block<unsigned long long>;

} // namespace spline
