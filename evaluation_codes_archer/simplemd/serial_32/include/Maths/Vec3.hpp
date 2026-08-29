#ifndef VEC3_HPP
#define VEC3_HPP

/**************************************************************************************************/
#include "../Helper/_helper.hpp"
/**************************************************************************************************/

/**
 * @file Vec3.hpp
 * @brief 3D vector class with common linear algebra operations.
 *
 * Provides a lightweight 3-component vector type intended for numerical
 * and scientific computing. Supports arithmetic, norms, dot/cross products,
 * periodic boundary conditions, and basis transformations.
 */

namespace Maths
{

/**
 * @class Vec3
 * @brief 3D Cartesian vector.
 *
 * Represents a three-dimensional vector with float precision components.
 * Designed for performance and clarity in scientific and numerical codes.
 */
class Vec3
{
public:

    //#######################################
    // Attributes
    //#######################################

    /** @brief x-component */
    float x {};

    /** @brief y-component */
    float y {};

    /** @brief z-component */
    float z {};

    //#######################################
    // Constructors
    //#######################################

    /**
     * @brief Default constructor.
     *
     * Initializes the vector to (0,0,0).
     */
    constexpr Vec3() : x(0), y(0), z(0) {}

    /**
     * @brief Construct from components.
     * @param x x-component
     * @param y y-component
     * @param z z-component
     */
    constexpr Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    //#######################################
    // Methods
    //#######################################

    /**
     * @brief Unary minus.
     * @return Vector with all components negated.
     */
    inline Vec3 operator-() const
    {
        return Vec3(-x, -y, -z);
    }

    /**
     * @brief Component access by index.
     * @param i Index (0=x, 1=y, 2=z)
     * @return Reference to component.
     * @warning No bounds checking is performed.
     */
    inline float& operator[](int i)
    {
        return *((&x) + i);
    }

    /**
     * @brief Component access by index (const).
     * @param i Index (0=x, 1=y, 2=z)
     * @return Const reference to component.
     * @warning No bounds checking is performed.
     */
    inline const float& operator[](int i) const
    {
        return *((&x) + i);
    }

    /**
     * @brief Euclidean norm of the vector.
     * @return Vector magnitude.
     */
    inline float length() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    /**
     * @brief Squared Euclidean norm.
     *
     * Useful for performance-critical code where the true length is not required.
     *
     * @return Squared magnitude.
     */
    inline float length_squared() const
    {
        return x * x + y * y + z * z;
    }

    /**
     * @brief Return a normalized copy of the vector.
     * @return Unit vector in the same direction, or the original vector if zero-length.
     */
    inline Vec3 normalize() const
    {
        float len = length();
        return len > 0 ? (*this) / len : *this;
    }

    /**
     * @brief Dot product with another vector.
     * @param other Other vector.
     * @return Scalar dot product.
     */
    inline float dot(const Vec3& other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    /**
     * @brief Cross product with another vector.
     * @param other Other vector.
     * @return Cross product vector.
     */
    inline Vec3 cross(const Vec3& other) const
    {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    /**
     * @brief Distance to another vector.
     * @param other Other vector.
     * @return Euclidean distance.
     */
    inline float distance(const Vec3& other) const
    {
        return (*this - other).length();
    }

    /**
     * @brief Clamp components to a range.
     * @param minVal Minimum value.
     * @param maxVal Maximum value.
     * @return Clamped vector.
     */
    inline Vec3 clamp(float minVal = 0.0f, float maxVal = 1.0f) const
    {
        return Vec3(
            std::clamp(x, minVal, maxVal),
            std::clamp(y, minVal, maxVal),
            std::clamp(z, minVal, maxVal)
        );
    }

    /**
     * @brief Apply periodic boundary conditions in a unit cell.
     *
     * Wraps all components into the interval [0,1).
     */
    void unit_cell_pbc()
    {
        x = std::fmod(x, 1.0f);
        y = std::fmod(y, 1.0f);
        z = std::fmod(z, 1.0f);
        if (x < 0.0f) x += 1.0f;
        if (y < 0.0f) y += 1.0f;
        if (z < 0.0f) z += 1.0f;
    }

    /**
     * @brief Add another vector (in-place).
     * @param other Vector to add.
     * @return Reference to this vector.
     */
    inline Vec3& operator+=(const Vec3& other)
    {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }

    /**
     * @brief Subtract another vector (in-place).
     * @param other Vector to subtract.
     * @return Reference to this vector.
     */
    inline Vec3& operator-=(const Vec3& other)
    {
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }

    /**
     * @brief Scale vector (in-place).
     * @param scalar Scaling factor.
     * @return Reference to this vector.
     */
    inline Vec3& operator*=(float scalar)
    {
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }

    /**
     * @brief Divide vector by scalar (in-place).
     * @param scalar Divisor.
     * @return Reference to this vector.
     * @throws std::invalid_argument if scalar is zero.
     */
    inline Vec3& operator/=(float scalar)
    {
        if (scalar == 0.0f) THROW_INVALID_ARGUMENT("Divide by zero error.");
        x /= scalar; y /= scalar; z /= scalar;
        return *this;
    }

    /**
     * @brief Set all components to zero.
     */
    void zero() noexcept
    {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }

    //#######################################
    // Static utilities
    //#######################################

    /**
     * @brief Component-wise minimum of two vectors.
     */
    inline static Vec3 min(const Vec3& a, const Vec3& b) noexcept
    {
        return Vec3(
            std::min(a.x, b.x),
            std::min(a.y, b.y),
            std::min(a.z, b.z)
        );
    }

    /**
     * @brief Component-wise maximum of two vectors.
     */
    inline static Vec3 max(const Vec3& a, const Vec3& b) noexcept
    {
        return Vec3(
            std::max(a.x, b.x),
            std::max(a.y, b.y),
            std::max(a.z, b.z)
        );
    }

    /**
     * @brief Separation vector in a general lattice basis.
     *
     * @param alat Lattice constant.
     * @param basis 3x3 basis matrix stored row-major.
     * @param a First position.
     * @param b Second position.
     * @return Separation vector in Cartesian coordinates.
     */
    [[nodiscard]]
    inline static Vec3 separation(
        float alat,
        const std::array<float, 9>& basis,
        const Vec3& a,
        const Vec3& b) noexcept
    {
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float dz = b.z - a.z;

        const auto M = [&](int r, int c) noexcept { return basis[r * 3 + c]; };

        const float s0 = M(0,0)*dx + M(0,1)*dy + M(0,2)*dz;
        const float s1 = M(1,0)*dx + M(1,1)*dy + M(1,2)*dz;
        const float s2 = M(2,0)*dx + M(2,1)*dy + M(2,2)*dz;

        return Vec3(alat * s0, alat * s1, alat * s2);
    }

    //#######################################
    // Friends
    //#######################################

    /**
     * @brief Stream output operator.
     */
    friend inline std::ostream& operator<<(std::ostream& os, const Vec3& v)
    {
        return os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    }

    /** @brief Vector addition */
    friend inline Vec3 operator+(const Vec3& a, const Vec3& b)
    {
        return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    /** @brief Vector subtraction */
    friend inline Vec3 operator-(const Vec3& a, const Vec3& b)
    {
        return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    /** @brief Scalar multiplication */
    friend inline Vec3 operator*(const Vec3& a, float s)
    {
        return Vec3(a.x * s, a.y * s, a.z * s);
    }

    /** @brief Scalar multiplication */
    friend inline Vec3 operator*(float s, const Vec3& a)
    {
        return a * s;
    }

    /** @brief Matrix-vector multiplication (C-style 3x3 matrix) */
    friend inline Vec3 operator*(const float (&m)[3][3], const Vec3& v)
    {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        };
    }

    /** @brief Matrix-vector multiplication (std::array basis) */
    friend inline Vec3 operator*(const std::array<float, 9>& basis, const Vec3& v)
    {
        return {
            basis[0]*v.x + basis[1]*v.y + basis[2]*v.z,
            basis[3]*v.x + basis[4]*v.y + basis[5]*v.z,
            basis[6]*v.x + basis[7]*v.y + basis[8]*v.z
        };
    }

    /** @brief Scalar division */
    friend inline Vec3 operator/(const Vec3& a, float s)
    {
        if (s == 0.0f) THROW_INVALID_ARGUMENT("Divide by zero error.");
        return Vec3(a.x / s, a.y / s, a.z / s);
    }
};

} // namespace Maths

#endif // VEC3_HPP
