#include "DualBlock.h"
#include "helpers.h"

namespace spline {

template <class T>
size_t DualBlock<T>::size() const {
#if !defined(__CUDA_ARCH__)
    return block_cpu_.size();
#else
    return block_cuda_.size();
#endif
}

template <class T>
size_t DualBlock<T>::byte_size() const {
#if !defined(__CUDA_ARCH__)
    return block_cpu_.size() * sizeof(T);
#else
    return block_cuda_.size() * sizeof(T);
#endif
}

template <class T>
const T &DualBlock<T>::operator[](size_t pos) const {
#if !defined(__CUDA_ARCH__)
    return block_cpu_[pos];
#else
    return block_cuda_[pos];
#endif
}

template <class T>
const T *DualBlock<T>::data() const noexcept {
#if !defined(__CUDA_ARCH__)
    return block_cpu_.data();
#else
    return block_cuda_.data();
#endif
}

template <class T>
const T *DualBlock<T>::begin() const noexcept {
#if !defined(__CUDA_ARCH__)
    return block_cpu_.data();
#else
    return block_cuda_.begin();
#endif
}

template <class T>
const T *DualBlock<T>::cbegin() const noexcept {
#if !defined(__CUDA_ARCH__)
    return block_cpu_.data();
#else
    return block_cuda_.cbegin();
#endif
}

template <class T>
const T *DualBlock<T>::end() const noexcept {
#if !defined(__CUDA_ARCH__)
    return block_cpu_.data() + block_cpu_.size();
#else
    return block_cuda_.end();
#endif
}

template <class T>
const T *DualBlock<T>::cend() const noexcept {
#if !defined(__CUDA_ARCH__)
    return block_cpu_.data() + block_cpu_.size();
#else
    return block_cuda_.cend();
#endif
}

template <class T>
void DualBlock<T>::sync() {
    if (!sync_) {
        block_cuda_.assign(block_cpu_);
        sync_ = true;
    }
}

template class DualBlock<double>;
template class DualBlock<float>;

} // namespace spline
