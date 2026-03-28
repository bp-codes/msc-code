#ifndef SYCL_FUNCTIONS_HPP
#define SYCL_FUNCTIONS_HPP

#include <cmath>
#include <algorithm>
#include <sycl/sycl.hpp>

namespace Maths
{

#ifdef FLOAT64

    // -----------------------------
    // exp
    // -----------------------------
    [[nodiscard]] inline double sycl_compatible_exp(double x)
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
    [[nodiscard]] inline double sycl_compatible_log(double x)
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
    [[nodiscard]] inline double sycl_compatible_sqrt(double x)
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
    [[nodiscard]] inline double sycl_compatible_fmod(double x, double y)
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
    [[nodiscard]] inline double sycl_compatible_max(double x, double y)
    {
    #ifdef __SYCL_DEVICE_ONLY__
        return sycl::max(x, y);
    #else
        return std::max(x, y);
    #endif
    }

    // -----------------------------
    // min
    // -----------------------------
    [[nodiscard]] inline double sycl_compatible_min(double x, double y)
    {
    #ifdef __SYCL_DEVICE_ONLY__
        return sycl::min(x, y);
    #else
        return std::min(x, y);
    #endif
    }

    // -----------------------------
    // clamp
    // -----------------------------
    [[nodiscard]] inline double sycl_compatible_clamp(double x, double lo, double hi)
    {
    #ifdef __SYCL_DEVICE_ONLY__
        return sycl::clamp(x, lo, hi);
    #else
        return std::clamp(x, lo, hi);
    #endif
    }

#else

    // -----------------------------
    // exp
    // -----------------------------
    [[nodiscard]] inline float sycl_compatible_exp(float x)
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
    [[nodiscard]] inline float sycl_compatible_log(float x)
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
    [[nodiscard]] inline float sycl_compatible_sqrt(float x)
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
    [[nodiscard]] inline float sycl_compatible_fmod(float x, float y)
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
    [[nodiscard]] inline float sycl_compatible_max(float x, float y)
    {
    #ifdef __SYCL_DEVICE_ONLY__
        return sycl::max(x, y);
    #else
        return std::max(x, y);
    #endif
    }

    // -----------------------------
    // min
    // -----------------------------
    [[nodiscard]] inline float sycl_compatible_min(float x, float y)
    {
    #ifdef __SYCL_DEVICE_ONLY__
        return sycl::min(x, y);
    #else
        return std::min(x, y);
    #endif
    }

    // -----------------------------
    // clamp
    // -----------------------------
    [[nodiscard]] inline float sycl_compatible_clamp(float x, float lo, float hi)
    {
    #ifdef __SYCL_DEVICE_ONLY__
        return sycl::clamp(x, lo, hi);
    #else
        return std::clamp(x, lo, hi);
    #endif
    }


#endif

}

#endif