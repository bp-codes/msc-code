#ifndef VEC3_HPP
#define VEC3_HPP

/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
#include "SyclFunctions.hpp"
/*********************************************************************************************************************************/
namespace Maths
{
  



class Vec3 
{

public:

    //#######################################
    // Attributes
    //#######################################

    float x {};
    float y {};
    float z {};


    //#######################################
    // Constructor
    //#######################################

    // Constructors
    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(float x, float y, float z) : x(x), y(y), z(z) {}



    //#######################################
    // Methods
    //#######################################

    // Unary minus
    inline Vec3 operator-() const 
    {
        return Vec3(-x, -y, -z);
    }

    // Indexing
    inline float& operator[](int i) 
    { 
        return *((&x) + i); 
    }
    inline const float& operator[](int i) const 
    { 
        return *((&x) + i); 
    }


    // Vector length
    inline float length() const 
    {
        return Maths::sycl_compatible_sqrt(x * x + y * y + z * z);
    }

    // Squared length (for performance)
    inline float length_squared() const 
    {
        return x * x + y * y + z * z;
    }

    // Normalize vector
    inline Vec3 normalize() const 
    {
        float len = length();
        return len > 0 ? (*this) / len : *this;
    }

    // Dot product
    inline float dot(const Vec3& other) const 
    {
        return x * other.x + y * other.y + z * other.z;
    }

    // Cross product
    inline Vec3 cross(const Vec3& other) const 
    {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    // Distance between two vectors
    inline float distance(const Vec3& other) const 
    {
        return (*this - other).length();
    }

    // Clamp components to [min, max]
    inline Vec3 clamp(float minVal = 0.0f, float maxVal = 1.0f) const 
    {
        return Vec3(
            Maths::sycl_compatible_clamp(x, minVal, maxVal),
            Maths::sycl_compatible_clamp(y, minVal, maxVal),
            Maths::sycl_compatible_clamp(z, minVal, maxVal)
        );
    }

    // Enforce periodic boundary condition in a unit cell
    void unit_cell_pbc()
    {
        x = Maths::sycl_compatible_fmod(x, 1.0f);
        y = Maths::sycl_compatible_fmod(y, 1.0f);
        z = Maths::sycl_compatible_fmod(z, 1.0f);
        if (x < 0.0f) x += 1.0f;
        if (y < 0.0f) y += 1.0f;
        if (z < 0.0f) z += 1.0f;
    }


    // Compound assignment
    inline Vec3& operator+=(const Vec3& other) 
    {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }

    inline Vec3& operator-=(const Vec3& other) 
    {
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }

    inline Vec3& operator*=(float scalar) 
    {
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }

    inline Vec3& operator/=(float scalar) 
    {
        if(scalar == 0.0f) THROW_INVALID_ARGUMENT("Divide by zero error.");
        x /= scalar; y /= scalar; z /= scalar;
        return *this;
    }


    void zero() noexcept
    { 
        x = 0.0f; 
        y = 0.0f; 
        z = 0.0f; 
    }


    //#######################################
    // Static
    //#######################################

    inline static Vec3 min(const Vec3& a, const Vec3& b) noexcept
    {
        return Vec3(
            Maths::sycl_compatible_min(a.x, b.x),
            Maths::sycl_compatible_min(a.y, b.y),
            Maths::sycl_compatible_min(a.z, b.z)
        );
    }

    inline static Vec3 max(const Vec3& a, const Vec3& b) noexcept
    {
        return Vec3(
            Maths::sycl_compatible_max(a.x, b.x),
            Maths::sycl_compatible_max(a.y, b.y),
            Maths::sycl_compatible_max(a.z, b.z)
        );
    }

    [[nodiscard]]
    inline static Vec3 separation(float alat, const std::array<float, 9>& basis, const Vec3& a, const Vec3& b) noexcept
    {
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float dz = b.z - a.z;

        // lambda to use as if a matrix
        const auto M = [&](int r, int c) noexcept { return basis[r*3 + c]; };

        const float s0 = M(0,0)*dx + M(0,1)*dy + M(0,2)*dz;
        const float s1 = M(1,0)*dx + M(1,1)*dy + M(1,2)*dz;
        const float s2 = M(2,0)*dx + M(2,1)*dy + M(2,2)*dz;

        const float aL = alat;
        return Vec3(aL*s0, aL*s1, aL*s2);
    }





    //#######################################
    // Friends
    //#######################################

    // Output stream
    friend inline std::ostream& operator<<(std::ostream& os, const Vec3& v) 
    {
        return os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    }

    // Arithmetic operators
    friend inline Vec3 operator+(const Vec3& a, const Vec3& b) 
    {
        return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    friend inline Vec3 operator-(const Vec3& a, const Vec3& b) 
    {
        return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    friend inline Vec3 operator*(const Vec3& a, float s) 
    {
        return Vec3(a.x * s, a.y * s, a.z * s);
    }

    friend inline Vec3 operator*(float s, const Vec3& a) 
    {
        return a * s;
    }

    friend inline Vec3 operator*(const float (&m)[3][3], const Vec3& v) 
    {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        };
    }

    friend inline Vec3 operator*(const std::array<float, 9>& basis, const Vec3& v) 
    {
        return {
            basis[0]*v.x + basis[1]*v.y + basis[2]*v.z,
            basis[3]*v.x + basis[4]*v.y + basis[5]*v.z,
            basis[6]*v.x + basis[7]*v.y + basis[8]*v.z
        };
    }

    friend inline Vec3 operator/(const Vec3& a, float s) 
    {
        if(s == 0.0) THROW_INVALID_ARGUMENT("Divide by zero error.");
        return Vec3(a.x / s, a.y / s, a.z / s);
    }


    

};


}
#endif // VEC3_HPP
