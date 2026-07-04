#ifndef HIP_FUNCTIONS_HPP
#define HIP_FUNCTIONS_HPP

#include <cmath>

#include <hip/hip_runtime.h>

namespace Maths {

// HIP device code can call the global C math overloads.  Keeping these wrappers
// means the rest of the solver can continue to call Maths::exp/fabs/fmax/etc.
template <typename T>
[[nodiscard]] __host__ __device__ inline T pow(T x, T y) {
    return ::pow(x, y);
}

template <typename T>
[[nodiscard]] __host__ __device__ inline T exp(T x) {
    return ::exp(x);
}

template <typename T>
[[nodiscard]] __host__ __device__ inline T log(T x) {
    return ::log(x);
}

template <typename T>
[[nodiscard]] __host__ __device__ inline T sqrt(T x) {
    return ::sqrt(x);
}

template <typename T>
[[nodiscard]] __host__ __device__ inline T fmod(T x, T y) {
    return ::fmod(x, y);
}

template <typename T>
[[nodiscard]] __host__ __device__ inline T fmax(T x, T y) {
    return ::fmax(x, y);
}

template <typename T>
[[nodiscard]] __host__ __device__ inline T fmin(T x, T y) {
    return ::fmin(x, y);
}

template <typename T>
[[nodiscard]] __host__ __device__ inline T clamp(T x, T lo, T hi) {
    return Maths::fmin(Maths::fmax(x, lo), hi);
}

template <typename T>
[[nodiscard]] __host__ __device__ inline T fabs(T x) {
    return ::fabs(x);
}

}  // namespace Maths

#endif  // HIP_FUNCTIONS_HPP
