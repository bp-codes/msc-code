#ifndef VEC3_HPP
#define VEC3_HPP

/*********************************************************************************************************************************/
#include "Helper/CudaHelper.hpp"
#include "Maths/MathsFunctions.hpp"
#include <cuda_runtime.h>

/*********************************************************************************************************************************/
namespace Maths {

struct Vec3 {
public:
    // #######################################
    //  Attributes
    // #######################################

    float x{};
    float y{};
    float z{};

    // #######################################
    //  Constructor
    // #######################################

    // Constructors
    SIMPLEMD_HOST_DEVICE
    Vec3() : x(0.0f), y(0.0f), z(0.0f) {}

    SIMPLEMD_HOST_DEVICE
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    // #######################################
    //  Methods
    // #######################################

    // Unary minus
    SIMPLEMD_HOST_DEVICE
    inline Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    // Indexing
    SIMPLEMD_HOST_DEVICE
    inline float& operator[](int i) {
        return *((&x) + i);
    }

    SIMPLEMD_HOST_DEVICE
    inline const float& operator[](int i) const {
        return *((&x) + i);
    }

    // Vector length
    SIMPLEMD_HOST_DEVICE
    inline float length() const {
        return Maths::sqrt(x * x + y * y + z * z);
    }

    // Squared length (for performance)
    SIMPLEMD_HOST_DEVICE
    inline float length_squared() const {
        return x * x + y * y + z * z;
    }

    // Normalize vector
    SIMPLEMD_HOST_DEVICE
    inline Vec3 normalize() const {
        float len = length();
        return len > 0 ? (*this) / len : *this;
    }

    // Dot product
    SIMPLEMD_HOST_DEVICE
    inline float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Cross product
    SIMPLEMD_HOST_DEVICE
    inline Vec3 cross(const Vec3& other) const {
        return Vec3(y * other.z - z * other.y, z * other.x - x * other.z,
                    x * other.y - y * other.x);
    }

    // Distance between two vectors
    SIMPLEMD_HOST_DEVICE
    inline float distance(const Vec3& other) const {
        return (*this - other).length();
    }

    // Clamp components to [min, max]
    SIMPLEMD_HOST_DEVICE
    inline Vec3 clamp(float minVal = 0.0f, float maxVal = 1.0f) const {
        return Vec3(Maths::clamp(x, minVal, maxVal), Maths::clamp(y, minVal, maxVal),
                    Maths::clamp(z, minVal, maxVal));
    }

    // Enforce periodic boundary condition in a unit cell
    SIMPLEMD_HOST_DEVICE
    void unit_cell_pbc() {
        x = Maths::fmod(x, 1.0f);
        y = Maths::fmod(y, 1.0f);
        z = Maths::fmod(z, 1.0f);
        if (x < 0.0f)
            x += 1.0f;
        if (y < 0.0f)
            y += 1.0f;
        if (z < 0.0f)
            z += 1.0f;
    }

    // Compound assignment
    SIMPLEMD_HOST_DEVICE
    inline Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    SIMPLEMD_HOST_DEVICE
    inline Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    SIMPLEMD_HOST_DEVICE
    inline Vec3& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    SIMPLEMD_HOST_DEVICE
    inline Vec3& operator/=(float scalar) {
        auto s_div = scalar;
        if (s_div == 0.0f)
            s_div = 1.0e-10f;
        //     THROW_INVALID_ARGUMENT("Divide by zero error.");
        x /= s_div;
        y /= s_div;
        z /= s_div;
        return *this;
    }

    SIMPLEMD_HOST_DEVICE
    void zero() noexcept {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }

    // #######################################
    //  Static
    // #######################################

    SIMPLEMD_HOST_DEVICE
    inline static Vec3 min(const Vec3& a, const Vec3& b) noexcept {
        return Vec3(Maths::fmin(a.x, b.x), Maths::fmin(a.y, b.y), Maths::fmin(a.z, b.z));
    }

    SIMPLEMD_HOST_DEVICE
    inline static Vec3 max(const Vec3& a, const Vec3& b) noexcept {
        return Vec3(Maths::fmax(a.x, b.x), Maths::fmax(a.y, b.y), Maths::fmax(a.z, b.z));
    }

    SIMPLEMD_HOST_DEVICE
    inline static Vec3 separation(
        float alat,
        const std::array<float,9>& basis,
        const Vec3& a,
        const Vec3& b) noexcept
    {
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float dz = b.z - a.z;

        const float s0 =
            basis[0] * dx +
            basis[1] * dy +
            basis[2] * dz;

        const float s1 =
            basis[3] * dx +
            basis[4] * dy +
            basis[5] * dz;

        const float s2 =
            basis[6] * dx +
            basis[7] * dy +
            basis[8] * dz;

        return Vec3(
            alat * s0,
            alat * s1,
            alat * s2);
    }

    // #######################################
    //  Friends
    // #######################################

    // Output stream
    SIMPLEMD_HOST_DEVICE
    friend inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
        return os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    }

    // Arithmetic operators
    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator+(const Vec3& a, const Vec3& b) {
        return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator-(const Vec3& a, const Vec3& b) {
        return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator*(const Vec3& a, float s) {
        return Vec3(a.x * s, a.y * s, a.z * s);
    }

    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator*(float s, const Vec3& a) {
        return a * s;
    }

    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator*(const float (&m)[3][3], const Vec3& v) {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }

    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator*(const std::array<float, 9>& basis, const Vec3& v) {
        return {basis[0] * v.x + basis[1] * v.y + basis[2] * v.z,
                basis[3] * v.x + basis[4] * v.y + basis[5] * v.z,
                basis[6] * v.x + basis[7] * v.y + basis[8] * v.z};
    }

    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator/(const Vec3& a, float s) {
        auto s_div = s;
        if (s_div == 0.0f)
            s_div = 1.0e-10f;
            // THROW_INVALID_ARGUMENT("Divide by zero error.");
        return Vec3(a.x / s_div, a.y / s_div, a.z / s_div);
    }
};

}  // namespace Maths
#endif  // VEC3_HPP
