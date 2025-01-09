#include "IntersectorKernels.h"
#include "helpers.h"
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>


namespace spline {

namespace {

template <class T>
__device__ cuda::std::array<T, 2> Derivative(const cuda::std::array<T, 2> &scale, const T &t) {
    T r0 = - 2 * M_PI * scale[0] * sin(2.0 * M_PI * t);
    T r1 = 2 * M_PI * scale[1] * cos(2.0 * M_PI * t);
    return {r0, r1};
}

template
__device__ cuda::std::array<double, 2> Derivative(const cuda::std::array<double, 2> &scale,
                                                  const double &t);

template <class T>
__device__ cuda::std::array<T, 2> Value(const cuda::std::array<T, 2> &scale,
                                        const cuda::std::array<T, 2> &shift,
                                        const T &t) {
    T r0 = scale[0] * cos(2.0 * M_PI * t) + shift[0];
    T r1 = scale[1] * sin(2.0 * M_PI * t) + shift[1];
    return {r0, r1};
}

template
__device__ cuda::std::array<double, 2> Value(const cuda::std::array<double, 2> &scale,
                                             const cuda::std::array<double, 2> &shift,
                                             const double &t);

template <class T, int Degree>
__device__ cuda::std::array<T, 2> Value(const T * __restrict__ knots,
                                        const T * __restrict__ P,
                                        int knots_size,
                                        const T &t) {
    auto points_count = knots_size - Degree - 1;
    if (t == knots[0]) {
        return {P[0], P[points_count]};
    }
    if (t == knots[knots_size - 1]) {
        return {P[points_count - 1], P[2 * points_count - 1]};
    }
    T N[Degree + 1];
    for (int i = 0; i <= Degree; ++i) {
        N[i] = (T)0;
    }
    int k = 0;
    for (int i = 1; i < knots_size - 1; ++i) {
        if (t >= knots[i] && t < knots[i + 1]) {
            k = i;
            break;
        }
    }
    // Calculate coeffs
    N[Degree] = (T)1;
    for (int d = 1; d <= Degree; ++d) {
        N[Degree - d] = (knots[k + 1] - t) * N[(Degree - d) + 1] /
                        (knots[k + 1] - knots[k - d + 1]);
        for (int i = k - d + 1; i < k; ++i) {
            N[i - k + Degree] = (t - knots[i]) * N[i - k + Degree] / (knots[i + d] - knots[i]) +
                (knots[i + d + 1] - t) * N[i - k + Degree + 1] / (knots[i + d + 1] - knots[i + 1]);
        }
        N[Degree] = (t - knots[k]) * N[Degree] / (knots[k + d] - knots[k]);
    }
    T r0 = (T)0, r1 = (T)0;
    for (int i = 0; i <= Degree; ++i) {
        r0 += N[i] * P[k - Degree + i];
        r1 += N[i] * P[k - Degree + i + points_count];
    }
    return {r0, r1};
}

template
__device__ cuda::std::array<double, 2> Value<double, 3>(const double * __restrict__ knots,
                                                        const double * __restrict__ P,
                                                        int knots_size,
                                                        const double &t);
template
__device__ cuda::std::array<double, 2> Value<double, 2>(const double * __restrict__ knots,
                                                        const double * __restrict__ P,
                                                        int knots_size,
                                                        const double &t);

template <class T>
__device__ T CurveL2N(int idx,
                      int split_count,
                      const CurveCuda<T> * __restrict__ c0,
                      const CurveCuda<T> * __restrict__ c1,
                      const T &dt0,
                      const T &dt1
                     ) {
    int i = idx / split_count;
    int j = idx - i * split_count;
    T argn0 = std::max((T)0, std::min((T)1, i * dt0));
    T argn1 = std::max((T)0, std::min((T)1, j * dt1));
    auto f0 = (c0->type == CurveCuda<T>::CurveType::ellipse ?
                   Value<T>(c0->ellipse.scale, c0->ellipse.shift, argn0) :
                   Value<T, 3>(c0->spline.knots, c0->spline.P, c0->spline.knots_size, argn0));
    auto f1 = (c1->type == CurveCuda<T>::CurveType::ellipse ?
                   Value<T>(c1->ellipse.scale, c1->ellipse.shift, argn1) :
                   Value<T, 3>(c1->spline.knots, c1->spline.P, c1->spline.knots_size, argn1));
    T df = f0[0] - f1[0];
    T result = df * df;
    df = f0[1] - f1[1];
    result += df * df;
    return result;
}

template <class T>
__global__ void GetSearchIndexKernel(unsigned long long * __restrict__ index_dist,
                                     const unsigned long long * __restrict__ last_index_dist,
                                     const CurveCuda<T> * __restrict__ c0,
                                     const CurveCuda<T> * __restrict__ c1,
                                     T dt0,
                                     T dt1,
                                     int total_count
                                    ) {
    constexpr int THREADS_COUNT = 128;
    __shared__ T dists[THREADS_COUNT];
    __shared__ int sidcs[THREADS_COUNT];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < total_count) {
        int split_count = ceil((T)1 / dt1);
        float ldist = (last_index_dist == nullptr ? 0.f : __int_as_float((int)(*last_index_dist)));
        int split_idx = threadIdx.x;
        dists[split_idx] = CurveL2N(idx, split_count, c0, c1, dt0, dt1);
        sidcs[split_idx] = idx;

        for (int i = 0; i < 7; ++i) {
            int shift = (int)1 << i;
            int mask = (shift << 1) - 1;
            int other_idx = split_idx + shift;
            __syncthreads();
            if ((split_idx & mask) == 0 &&
                    other_idx < THREADS_COUNT &&
                    (other_idx + blockDim.x * blockIdx.x) < total_count &&
                    (dists[other_idx] <= dists[split_idx] || dists[split_idx] <= ldist) &&
                    dists[other_idx] > ldist) {
                dists[split_idx] = dists[other_idx];
                sidcs[split_idx] = sidcs[other_idx];
            }
        }
        __syncthreads();
        if (split_idx == 0) {
            unsigned long long cur = *index_dist;
            unsigned long long val = ((unsigned long long)sidcs[0] << 32) |
                                      (unsigned long long)__float_as_int(dists[0]);
            while ((int)cur == 0 || __int_as_float((int)cur) > dists[0]) {
                unsigned long long actual = atomicCAS(index_dist, cur, val);
                if (actual == cur) {
                    break;
                } else {
                    cur = actual;
                }
            }

        }
    }
}

template <class T>
__global__ void IndexToPointsKernel(unsigned long long * __restrict__ index,
                                    T * __restrict__ result,
                                    T dt0,
                                    T dt1,
                                    int dim,
                                    int total_count
                                   ) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < total_count) {
        int result_idx = (int)(index[idx] >> 32);
        int split_count = ceil((T)1 / dt1);
        int i = result_idx / split_count;
        int j = result_idx - i * split_count;
        result[idx * dim] = std::max((T)0, std::min((T)1, i * dt0));
        result[idx * dim + 1] = std::max((T)0, std::min((T)1, j * dt1));
    }
}

} //unnamed namespace

template <class T, class Curve0, class Curve1>
void CudaGetSearchPoints(Block<T> &results_cuda,
                         Curve0 &c0,
                         Curve1 &c1,
                         T ratio,
                         T select_ratio,
                         T min_dparam,
                         T max_dparam
                        ) {
    if (ratio <= (T)0 || select_ratio <= (T)0 || min_dparam > max_dparam ||
        min_dparam <= (T)0 || max_dparam >= (T)1) {
        CUDA_GENERAL_ERROR("invalid input");
    }
    size_t num_tries = (T)1 / (std::min(c0.GetMinDParam(),
                                        c1.GetMinDParam())) * select_ratio;
    results_cuda.resize(num_tries * 2, 0);
    unsigned long long *indices_dists = reinterpret_cast<unsigned long long *>(results_cuda.data());
    size_t dim = 2;
    c0.FillCudaBufferFloat();
    c1.FillCudaBufferFloat();
    float dt0f = c0.GetMinDParam() * ratio;
    float dt1f = c1.GetMinDParam() * ratio;
    dt0f = std::min((float)max_dparam, std::max((float)min_dparam, dt0f));
    dt1f = std::min((float)max_dparam, std::max((float)min_dparam, dt1f));
    size_t total_count = std::ceil(1.f / dt0f) * std::ceil(1.f / dt1f);
    for (int i = 0; i < num_tries; ++i) {
        KernelGrid sgrid(total_count, 128);
        GetSearchIndexKernel<float><<<sgrid.gsize(), sgrid.bsize()>>>(indices_dists + i,
                                                                     (i == 0 ? nullptr :
                                                                      indices_dists + i - 1),
                                                                     c0.GetCudaBufferFloat(),
                                                                     c1.GetCudaBufferFloat(),
                                                                     dt0f,
                                                                     dt1f,
                                                                     total_count
                                                                    );
    }
    T dt0 = c0.GetMinDParam() * ratio;
    T dt1 = c1.GetMinDParam() * ratio;
    dt0 = std::min((T)max_dparam, std::max((T)min_dparam, dt0));
    dt1 = std::min((T)max_dparam, std::max((T)min_dparam, dt1));
    KernelGrid sgrid(num_tries);
    IndexToPointsKernel<T><<<sgrid.gsize(), sgrid.bsize()>>>(indices_dists,
                                                             results_cuda.data(),
                                                             dt0,
                                                             dt1,
                                                             dim,
                                                             num_tries
                                                            );
}

namespace {

template <class T>
__device__ T CalcArg(const T &arg, const T &delta, int idx, int power_split) {
    T result = arg - delta + (T)2 * idx * delta / power_split;
    return std::max((T)0, std::min((T)1, result));
}

template <class T>
__global__ void OptimizeBruteForceKernel(double * __restrict__ results,
                                         const CurveCuda<T> * __restrict__ c0,
                                         const CurveCuda<T> * __restrict__ c1,
                                         int power_split,
                                         T delta0,
                                         T delta1,
                                         int dim,
                                         int total_count
                                        ) {
    __shared__ T dists[128];
    __shared__ int sidcs[128];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < total_count) {
        int ridx = blockIdx.x;
        T arg0 = results[ridx * dim];
        T arg1 = results[ridx * dim + 1];
        int split_idx = threadIdx.x;
        if (split_idx < power_split * power_split) {
            int i = split_idx / power_split;
            int j = split_idx - i * power_split;
            T argn0 = CalcArg(arg0, delta0, i, power_split);
            T argn1 = CalcArg(arg1, delta1, j, power_split);
            auto f0 = (c0->type == CurveCuda<T>::CurveType::ellipse ?
                       Value<T>(c0->ellipse.scale, c0->ellipse.shift, argn0) :
                       Value<T, 3>(c0->spline.knots, c0->spline.P, c0->spline.knots_size, argn0));
            auto f1 = (c1->type == CurveCuda<T>::CurveType::ellipse ?
                       Value<T>(c1->ellipse.scale, c1->ellipse.shift, argn1) :
                       Value<T, 3>(c1->spline.knots, c1->spline.P, c1->spline.knots_size, argn1));
            T df = f0[0] - f1[0];
            dists[split_idx] = df * df;
            df = f0[1] - f1[1];
            dists[split_idx] += df * df;
            sidcs[split_idx] = split_idx;

            for (int i = 0; i < 7; ++i) {
                int shift = (int)1 << i;
                int mask = (shift << 1) - 1;
                int other_idx = split_idx + shift;
                __syncthreads();
                if ((split_idx & mask) == 0 &&
                    other_idx < power_split * power_split &&
                    dists[other_idx] <= dists[split_idx]) {
                        dists[split_idx] = dists[other_idx];
                        sidcs[split_idx] = sidcs[other_idx];
                }
            }
            __syncthreads();
            if (split_idx == 0) {
                int i = sidcs[0] / power_split;
                int j = sidcs[0] - i * power_split;
                results[ridx * dim] = CalcArg(arg0, delta0, i, power_split);
                results[ridx * dim + 1] = CalcArg(arg1, delta1, j, power_split);
            }
        }
    }
}

} //unnamed namespace

template <class T, class Curve0, class Curve1>
std::vector<std::vector<T>> CudaOptimize(Block <T> &results_cuda,
                                         Curve0 &c0,
                                         Curve1 &c1,
                                         size_t max_iters,
                                         T epsilon,
                                         size_t power_split,
                                         T power_step,
                                         T ratio
                                        ) {
    if (!results_cuda.size() || epsilon <= (T)0 || power_step < (T)1 || power_split < 1) {
        CUDA_GENERAL_ERROR("invalid input");
    }
    size_t dim = 2;
    c0.FillCudaBuffer();
    c1.FillCudaBuffer();
    if (power_split > 1 && power_step > (T)1) {
        T delta0 = c0.GetMinDParam() * ratio;
        T delta1 = c1.GetMinDParam() * ratio;
        size_t float_iters_count = std::ceil(std::log((T)1e3) / std::log(power_step));
        size_t iters_count = std::ceil(std::log((T)1.0/epsilon) / std::log(power_step));
        KernelGrid grid(results_cuda.size() * 64, 128);
        for (size_t i = 0; i < float_iters_count; ++i) {
            OptimizeBruteForceKernel<float><<<grid.gsize(), grid.bsize()>>>(results_cuda.data(),
                                                                            c0.GetCudaBufferFloat(),
                                                                            c1.GetCudaBufferFloat(),
                                                                            power_split,
                                                                            delta0,
                                                                            delta1,
                                                                            dim,
                                                                            results_cuda.size() * 64
                                                                           );
            delta0 /= power_step;
            delta1 /= power_step;
        }
        delta0 *= power_step;
        delta1 *= power_step;
        for (size_t i = 0; i < iters_count - float_iters_count + 1; ++i) {
            OptimizeBruteForceKernel<T><<<grid.gsize(), grid.bsize()>>>(results_cuda.data(),
                                                                        c0.GetCudaBuffer(),
                                                                        c1.GetCudaBuffer(),
                                                                        power_split,
                                                                        delta0,
                                                                        delta1,
                                                                        dim,
                                                                        results_cuda.size() * 64
                                                                       );
            delta0 /= power_step;
            delta1 /= power_step;
        }
    }
    auto unflatten = [](const std::vector<T> &v2d, size_t dim) -> std::vector<std::vector<T>>{
        std::vector<std::vector<T>> unflat(v2d.size() / dim);
        for (size_t i = 0; i < v2d.size(); ++i) {
            size_t idx = i / dim;
            unflat[idx].push_back(v2d[i]);
        }
        return unflat;
    };
    auto results = unflatten(results_cuda.to_vector(), dim);
    return results;
}

template std::vector<std::vector<double>> CudaOptimize<double, SplineD<double, 3>, SplineD<double, 3>>(
    Block<double> &,
    SplineD<double, 3> &,
    SplineD<double, 3> &,
    size_t,
    double,
    size_t,
    double,
    double
   );

template std::vector<std::vector<double>> CudaOptimize<double, Ellipse<double>, SplineD<double, 3>>(
    Block<double> &,
    Ellipse<double> &,
    SplineD<double, 3> &,
    size_t,
    double,
    size_t,
    double,
    double
   );

template std::vector<std::vector<double>> CudaOptimize<double, SplineD<double, 3>, Ellipse<double>>(
    Block<double> &,
    SplineD<double, 3> &,
    Ellipse<double> &,
    size_t,
    double,
    size_t,
    double,
    double
   );

template std::vector<std::vector<double>> CudaOptimize<double, Ellipse<double>, Ellipse<double>>(
    Block<double> &,
    Ellipse<double> &,
    Ellipse<double> &,
    size_t,
    double,
    size_t,
    double,
    double
   );

template void
CudaGetSearchPoints<double, Ellipse<double>, Ellipse<double>>(
    Block<double> &,
    Ellipse<double> &,
    Ellipse<double> &,
    double,
    double,
    double,
    double
    );

template void
CudaGetSearchPoints<double, Ellipse<double>, SplineD<double, 3>>(
    Block<double> &,
    Ellipse<double> &,
    SplineD<double, 3> &,
    double,
    double,
    double,
    double
    );

template void
CudaGetSearchPoints<double, SplineD<double, 3>, Ellipse<double>>(
    Block<double> &,
    SplineD<double, 3> &,
    Ellipse<double> &,
    double,
    double,
    double,
    double
    );

template void
CudaGetSearchPoints<double, SplineD<double, 3>, SplineD<double, 3>>(
    Block<double> &,
    SplineD<double, 3> &,
    SplineD<double, 3> &,
    double,
    double,
    double,
    double
    );

} // namespace spline
