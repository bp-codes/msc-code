#ifndef MATH_FUNCTIONS_HPP
#define MATH_FUNCTIONS_HPP

#include <cmath>
#include <concepts>

namespace Maths {

// -----------------------------
// pow
// -----------------------------
template <typename T>
[[nodiscard]] inline T pow(T x, T y) {
    return std::pow(x, y);
}

// -----------------------------
// exp
// -----------------------------
template <typename T>
[[nodiscard]] inline T exp(T x) {
    return std::exp(x);
}

// -----------------------------
// log
// -----------------------------
template <typename T>
[[nodiscard]] inline T log(T x) {
    return std::log(x);
}

// -----------------------------
// sqrt
// -----------------------------
template <typename T>
[[nodiscard]] inline T sqrt(T x) {
    return std::sqrt(x);
}

// -----------------------------
// fmod
// -----------------------------
template <typename T>
[[nodiscard]] inline T fmod(T x, T y) {
    return std::fmod(x, y);
}

// -----------------------------
// max
// -----------------------------
template <typename T>
[[nodiscard]] inline T fmax(T x, T y) {
    return std::fmax(x, y);
}

// -----------------------------
// min
// -----------------------------
template <typename T>
[[nodiscard]] inline T fmin(T x, T y) {
    return std::fmin(x, y);
}

// -----------------------------
// clamp
// -----------------------------
template <typename T>
[[nodiscard]] inline T clamp(T x, T lo, T hi) {
    return std::fmin(std::fmax(x, lo), hi);
}

}  // namespace Maths

#endif
