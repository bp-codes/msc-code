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

    double x{};
    double y{};
    double z{};

    // #######################################
    //  Constructor
    // #######################################

    // Constructors
    SIMPLEMD_HOST_DEVICE
    Vec3() : x(0), y(0), z(0) {}

    SIMPLEMD_HOST_DEVICE
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

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
    inline double& operator[](int i) {
        return *((&x) + i);
    }

    SIMPLEMD_HOST_DEVICE
    inline const double& operator[](int i) const {
        return *((&x) + i);
    }

    // Vector length
    SIMPLEMD_HOST_DEVICE
    inline double length() const {
        return Maths::sqrt(x * x + y * y + z * z);
    }

    // Squared length (for performance)
    SIMPLEMD_HOST_DEVICE
    inline double length_squared() const {
        return x * x + y * y + z * z;
    }

    // Normalize vector
    SIMPLEMD_HOST_DEVICE
    inline Vec3 normalize() const {
        double len = length();
        return len > 0 ? (*this) / len : *this;
    }

    // Dot product
    SIMPLEMD_HOST_DEVICE
    inline double dot(const Vec3& other) const {
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
    inline double distance(const Vec3& other) const {
        return (*this - other).length();
    }

    // Clamp components to [min, max]
    SIMPLEMD_HOST_DEVICE
    inline Vec3 clamp(double minVal = 0.0, double maxVal = 1.0) const {
        return Vec3(Maths::clamp(x, minVal, maxVal), Maths::clamp(y, minVal, maxVal),
                    Maths::clamp(z, minVal, maxVal));
    }

    // Enforce periodic boundary condition in a unit cell
    SIMPLEMD_HOST_DEVICE
    void unit_cell_pbc() {
        x = Maths::fmod(x, 1.0);
        y = Maths::fmod(y, 1.0);
        z = Maths::fmod(z, 1.0);
        if (x < 0.0)
            x += 1.0;
        if (y < 0.0)
            y += 1.0;
        if (z < 0.0)
            z += 1.0;
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
    inline Vec3& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    SIMPLEMD_HOST_DEVICE
    inline Vec3& operator/=(double scalar) {
        auto s_div = scalar;
        if (s_div == 0.0)
            s_div = 1.0e-10;
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
        double alat,
        const std::array<double,9>& basis,
        const Vec3& a,
        const Vec3& b) noexcept
    {
        const double dx = b.x - a.x;
        const double dy = b.y - a.y;
        const double dz = b.z - a.z;

        const double s0 =
            basis[0] * dx +
            basis[1] * dy +
            basis[2] * dz;

        const double s1 =
            basis[3] * dx +
            basis[4] * dy +
            basis[5] * dz;

        const double s2 =
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
    friend inline Vec3 operator*(const Vec3& a, double s) {
        return Vec3(a.x * s, a.y * s, a.z * s);
    }

    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator*(double s, const Vec3& a) {
        return a * s;
    }

    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator*(const double (&m)[3][3], const Vec3& v) {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }

    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator*(const std::array<double, 9>& basis, const Vec3& v) {
        return {basis[0] * v.x + basis[1] * v.y + basis[2] * v.z,
                basis[3] * v.x + basis[4] * v.y + basis[5] * v.z,
                basis[6] * v.x + basis[7] * v.y + basis[8] * v.z};
    }

    SIMPLEMD_HOST_DEVICE
    friend inline Vec3 operator/(const Vec3& a, double s) {
        auto s_div = s;
        if (s_div == 0.0)
            s_div = 1.0e-10;
            // THROW_INVALID_ARGUMENT("Divide by zero error.");
        return Vec3(a.x / s_div, a.y / s_div, a.z / s_div);
    }
};

}  // namespace Maths
#endif  // VEC3_HPP
