#ifndef SYCL_FUNCTIONS_HPP
#define SYCL_FUNCTIONS_HPP

#include <cmath>
#include <concepts>
#include <sycl/sycl.hpp>

namespace SyclFunctions
{

// -----------------------------
// pow
// -----------------------------
template<std::floating_point T>
[[nodiscard]] inline T pow(T x, T y)
{
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::pow(x, y);
#else
    return std::pow(x, y);
#endif
}

// -----------------------------
// exp
// -----------------------------
template<std::floating_point T>
[[nodiscard]] inline T exp(T x)
{
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::exp(x);
#else
    return std::exp(x);
#endif
}

// -----------------------------
// log
// -----------------------------
template<std::floating_point T>
[[nodiscard]] inline T log(T x)
{
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::log(x);
#else
    return std::log(x);
#endif
}

// -----------------------------
// sqrt
// -----------------------------
template<std::floating_point T>
[[nodiscard]] inline T sqrt(T x)
{
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

// -----------------------------
// fmod
// -----------------------------
template<std::floating_point T>
[[nodiscard]] inline T fmod(T x, T y)
{
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fmod(x, y);
#else
    return std::fmod(x, y);
#endif
}

// -----------------------------
// max
// -----------------------------
template<std::floating_point T>
[[nodiscard]] inline T fmax(T x, T y)
{
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fmax(x, y);
#else
    return std::fmax(x, y);
#endif
}

// -----------------------------
// min
// -----------------------------
template<std::floating_point T>
[[nodiscard]] inline T fmin(T x, T y)
{
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fmin(x, y);
#else
    return std::fmin(x, y);
#endif
}

// -----------------------------
// clamp
// -----------------------------
template<std::floating_point T>
[[nodiscard]] inline T clamp(T x, T lo, T hi)
{
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fmin(sycl::fmax(x, lo), hi);
#else
    return std::fmin(std::fmax(x, lo), hi);
#endif
}

}

#endif