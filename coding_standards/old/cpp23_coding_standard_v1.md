# C++23 Coding Standard (Scientific / HPC)

## 1) Scope and goals
- **MUST** follow this standard for all new C++ code and when touching existing code in the same area.  
  *Rationale:* preserves consistency and reviewer expectations.
- **MUST** prioritize: correctness → performance/scalability → readability → consistency.  
  *Rationale:* HPC codebases live or die by correctness + performance.
- **SHOULD** keep APIs simple, explicit, and predictable (especially for numerical kernels).  
  *Rationale:* reduces bugs and helps compiler optimization.

---

## 2) File conventions and layout
- **MUST** use file extensions: `.cpp`, `.hpp`, `.ixx`.  
  *Rationale:* canonical project setting.
- **MUST** use include guards (`#ifndef/#define/#endif`) and **MUST NOT** use `#pragma once`.  
  *Rationale:* canonical project rule and matches examples.
- **MUST** name include guards as `UPPERCASE_FILENAME_HPP` (or `_IXX` if needed). *(Inferred from examples.)*  
  Example:
  ```cpp
  #ifndef VEC3_HPP
  #define VEC3_HPP
  // ...
  #endif
  ```
- **SHOULD** place a project-local include banner/separator consistently. *(Inferred from examples.)*

---

## 3) Formatting and braces
- **MUST** indent with **4 spaces**, **no tabs**.  
  *Rationale:* canonical setting.
- **MUST** use **Allman / BSD** braces.  
  Example:
  ```cpp
  void f()
  {
      // ...
  }
  ```
- **MUST** keep line breaks and indentation readable around long expressions and argument lists.  
  *Rationale:* maintainability + reviewability.
- **SHOULD** aim for functions that “fit on a screen” (roughly ≤ ~60–80 lines).  
  *Rationale:* encourages cohesion and testability.

---

## 4) Naming conventions
- **MUST** use:
  - variables: `snake_case` (`my_integer`)
  - types/classes/structs: `PascalCase` (`MyClass`)
  - macro-like parameters: `UPPERCASE` (`MYPARAMETER`)
  - constants: `UPPERCASE_WITH_UNDERSCORES` (`MY_CONST_INTEGER`)
- **SHOULD** use descriptive names (avoid single-letter except loop indices in tight loops).  
  *Rationale:* scientific code benefits from clarity.

---

## 5) Includes, headers, and modules
- **MUST** include what you use in headers (do not rely on transitive includes).  
  *Rationale:* reduces build fragility.
- **SHOULD** minimize header includes; prefer forward declarations when feasible (without complicating ownership).  
  *Rationale:* compile-time reduction.
- **MAY** use `.ixx` modules where appropriate; keep module boundaries aligned with library namespaces and stable APIs.  
  *Rationale:* improves build scaling (when adopted).
- **SHOULD** keep header-only utilities `inline`/`constexpr` as appropriate. *(Inferred from examples.)*

---

## 6) Namespaces and symbol visibility
- **MUST** use namespaces for libraries and logical subsystems (e.g., `Maths`, `SimpleMD`).  
  *Rationale:* prevents global pollution, improves organization.
- **MUST NOT** use `using namespace ...` at file scope.  
  *Rationale:* avoids symbol collisions and unclear provenance.
- **MUST NOT** use `using std::foo;` at file scope unless explicitly allowed by project policy (default: not allowed).  
  *Rationale:* same as above; “full naming only”.
- **SHOULD** close namespaces with a trailing comment. *(Inferred from examples.)*  
  Example:
  ```cpp
  } // namespace Maths
  ```

---

## 7) Types, initialization, and auto usage
- **MUST** prefer `std::size_t` over `size_t`.  
  *Rationale:* canonical rule; avoids ambiguity.
- **MUST** prefer brace initialization `{}` and use empty/zero-brace initialization where appropriate.  
  *Rationale:* consistent, avoids narrowing, matches examples.
- **SHOULD** prefer `auto` for local variables when it improves readability and the type is obvious from RHS.  
  Example:
  ```cpp
  auto n {std::size_t(numbers_a.size())};
  auto my_integer {0};
  auto my_double {0.0};
  auto my_counter {std::size_t(0)};
  auto my_string {std::string("")};
  ```
- **MUST** avoid `auto` when it obscures important types (e.g., iterator category, precision, signedness).  
  *Rationale:* numerical correctness and clarity.
- **SHOULD** mark value-returning helpers as `[[nodiscard]]` when ignoring the result is likely a bug. *(Inferred from examples.)*

---

## 8) Ownership, lifetime, and pointers
- **MUST** avoid raw owning pointers. Use `std::unique_ptr` / `std::shared_ptr` for ownership only.  
  *Rationale:* canonical rule; prevents leaks and unclear ownership.
- **MAY** use raw pointers only as **non-owning observers** when references/spans/iterators are unsuitable.  
  *Rationale:* allows interop and low-level APIs while preserving ownership clarity.
- **SHOULD** prefer references for required non-null inputs; prefer `std::span` for array-like views.  
  *Rationale:* explicit intent, easier optimization.
- **SHOULD** avoid unnecessary heap allocation in hot paths.  
  *Rationale:* HPC performance.
- **MUST** use `std::variant` over polymorphism by default; use polymorphism only when runtime extensibility or type erasure is clearly beneficial.  
  *Rationale:* canonical rule; variant enables value semantics and avoids vtables.

---

## 9) Control flow guidelines (if/switch/enums)
- **SHOULD** prefer `switch` over long `if/else if` chains when branching on the same discrete value.  
  *Rationale:* readability and compiler friendliness.
- **SHOULD** use `enum class` over `enum`.  
  *Rationale:* scoped enums avoid collisions and implicit conversions.
- **SHOULD** use enums for “mode” / “kind” settings rather than magic integers/strings.  
  *Rationale:* correctness and maintainability.
- **MUST** handle all enum values in switches (include `default` only when truly intended).  
  *Rationale:* prevents silent missing cases.
- **MUST** prefer `i++` over `++i` (unless necessary).  
  *Rationale:* canonical preference.
- **SHOULD** keep loop bounds and types consistent (especially signed/unsigned).  
  Example:
  ```cpp
  for (auto i = std::size_t(0); i < n; i++)
  {
      // ...
  }
  ```

---

## 10) STL usage and algorithms/iterators
- **MUST** use the STL where available instead of rewriting common utilities.  
  *Rationale:* correctness, portability, maintainability.
- **SHOULD** prefer standard algorithms and iterators where they improve clarity.  
  *Rationale:* expresses intent and reduces bugs.
- **MAY** write explicit loops in performance-critical sections when it improves vectorization or clarity.  
  *Rationale:* HPC kernels often benefit from explicit loops.
- **SHOULD** use `std::clamp`, `std::min`, `std::max`, etc. (as shown). *(Inferred from examples.)*

---

## 11) Error handling and contracts (assertions/exceptions)
- **MUST** use project error macros for throwing exceptions with context:
  - `THROW_INVALID_ARGUMENT(msg)`
  - `THROW_RUNTIME_ERROR(msg)`  
  *Rationale:* consistent, provides file/line/function context (examples).
- **SHOULD** throw `std::invalid_argument` for precondition violations at API boundaries.  
  *Rationale:* predictable error taxonomy.
- **MUST** avoid exceptions inside hot inner loops unless unavoidable; validate inputs before entering kernels.  
  *Rationale:* performance and control flow predictability.
- **SHOULD** mark non-throwing functions `noexcept` where correct. *(Inferred from examples.)*

---

## 12) Parallelism (OpenMP)
- **MUST** use OpenMP for parallel sections where appropriate.  
  *Rationale:* canonical rule and matches example usage.
- **SHOULD** keep OpenMP pragmas tightly scoped to the smallest loop/region needed.  
  *Rationale:* avoids unintended sharing and makes correctness reviewable.
- **MUST** ensure container sizes match and outputs are sized before the parallel region.  
  *Rationale:* avoids races/out-of-bounds in parallel loops.
- **SHOULD** prefer `schedule(static)` for uniform work in tight numerical loops unless profiling shows otherwise. *(Inferred from example.)*
- **MAY** use `simd` when it improves vectorization and correctness is maintained. *(Inferred from example.)*  
  Example:
  ```cpp
  #pragma omp parallel for simd schedule(static)
  for (auto i = std::size_t(0); i < n; i++)
  {
      numbers_c[i] = numbers_a[i] + numbers_b[i];
  }
  ```
- **MUST** avoid hidden data sharing: explicitly manage `shared`/`private` if the loop body becomes non-trivial.  
  *Rationale:* prevents subtle race bugs.

---

## 13) Documentation (Doxygen)
- **MUST** write Doxygen comments for public APIs (types, public methods, key free functions).  
  *Rationale:* canonical requirement; matches examples.
- **SHOULD** include:
  - `@file` + `@brief` in headers
  - `@param` for each parameter
  - `@return` for non-void
  - `@throws` when exceptions can occur
  - `@warning` for hazards (bounds checking, UB, etc.)  
  *Rationale:* consistent API documentation.
- **SHOULD** use `[[nodiscard]]` for important return values and document behavior for edge cases.  
  *Rationale:* correctness.

---

## 14) Forbidden patterns and preferred alternatives
- **MUST NOT** use raw owning pointers.  
  **Prefer:** `std::unique_ptr`, `std::shared_ptr`.
- **MUST NOT** use `#pragma once`.  
  **Prefer:** include guards.
- **MUST NOT** use `using namespace ...` at file scope.  
  **Prefer:** fully qualified names.
- **MUST NOT** use `std::endl` for normal output.  
  **Prefer:** `'\n'`.
- **SHOULD NOT** write long `if/else` ladders on discrete modes.  
  **Prefer:** `enum class` + `switch`.
- **SHOULD NOT** reimplement STL utilities.  
  **Prefer:** `<algorithm>`, `<numeric>`, `<ranges>` when appropriate.
- **SHOULD** avoid unnecessary polymorphism.  
  **Prefer:** `std::variant` and value semantics by default.

### Macros
- **MAY** use project-provided macros (`CLASS_SET_GET`, `SINGLETON`, etc.) only when they improve consistency and do not obscure control flow. *(Inferred from examples.)*
- **SHOULD** keep macros:
  - well-indented
  - parenthesized appropriately
  - used primarily for boilerplate reduction (getters/setters, standardized errors)
- **SHOULD** prefer `inline`/`constexpr` functions or templates over macros when type safety and debugging matter.  
  *Rationale:* macros can hide errors and hinder tooling.

---

## 15) Reviewer checklist
- Naming matches conventions (snake_case vars, PascalCase types, UPPERCASE constants/macros).
- 4 spaces, Allman braces, no tabs.
- Include guards present; no `#pragma once`.
- No `using namespace` at file scope; namespaces used appropriately.
- Initialization uses `{}`; `std::size_t` used for sizes/indices.
- `auto` used where types are obvious; not used where it obscures meaning.
- No raw owning pointers; ownership is explicit via smart pointers.
- STL used instead of custom reimplementations; algorithms/iterators used where sensible.
- `enum class` + `switch` used for modes; cases complete.
- Exceptions/macros used consistently (`THROW_INVALID_ARGUMENT`, `THROW_RUNTIME_ERROR`); no exceptions in hot loops.
- OpenMP pragmas are minimal-scope; loop correctness/race-safety checked.
- Public APIs have Doxygen docs (`@param`, `@return`, `@throws`, warnings where needed).

---

## Exceptions
- **Non-owning raw pointers** are **MAY**-allowed only for interoperability or low-level APIs where references/spans/iterators are impractical; ownership must remain elsewhere and be documented.
- **Explicit loops** may be preferred over STL algorithms in performance-critical kernels when it improves vectorization or clarity of the numerical operation.
- **Polymorphism** is permitted when runtime extension or plugin-style architecture is required and `std::variant` would be awkward or inefficient.

---

# Appendix: Canonical Examples

## A.1 Vec3 example (canonical)
```cpp
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
 * Represents a three-dimensional vector with double precision components.
 * Designed for performance and clarity in scientific and numerical codes.
 */
class Vec3
{
public:

    //#######################################
    // Attributes
    //#######################################

    /** @brief x-component */
    double x {};

    /** @brief y-component */
    double y {};

    /** @brief z-component */
    double z {};

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
    constexpr Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

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
    inline double& operator[](int i)
    {
        return *((&x) + i);
    }

    /**
     * @brief Component access by index (const).
     * @param i Index (0=x, 1=y, 2=z)
     * @return Const reference to component.
     * @warning No bounds checking is performed.
     */
    inline const double& operator[](int i) const
    {
        return *((&x) + i);
    }

    /**
     * @brief Euclidean norm of the vector.
     * @return Vector magnitude.
     */
    inline double length() const
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
    inline double length_squared() const
    {
        return x * x + y * y + z * z;
    }

    /**
     * @brief Return a normalized copy of the vector.
     * @return Unit vector in the same direction, or the original vector if zero-length.
     */
    inline Vec3 normalize() const
    {
        double len = length();
        return len > 0 ? (*this) / len : *this;
    }

    /**
     * @brief Dot product with another vector.
     * @param other Other vector.
     * @return Scalar dot product.
     */
    inline double dot(const Vec3& other) const
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
    inline double distance(const Vec3& other) const
    {
        return (*this - other).length();
    }

    /**
     * @brief Clamp components to a range.
     * @param minVal Minimum value.
     * @param maxVal Maximum value.
     * @return Clamped vector.
     */
    inline Vec3 clamp(double minVal = 0.0, double maxVal = 1.0) const
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
        x = std::fmod(x, 1.0);
        y = std::fmod(y, 1.0);
        z = std::fmod(z, 1.0);
        if (x < 0.0) x += 1.0;
        if (y < 0.0) y += 1.0;
        if (z < 0.0) z += 1.0;
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
    inline Vec3& operator*=(double scalar)
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
    inline Vec3& operator/=(double scalar)
    {
        if (scalar == 0.0) THROW_INVALID_ARGUMENT("Divide by zero error.");
        x /= scalar; y /= scalar; z /= scalar;
        return *this;
    }

    /**
     * @brief Set all components to zero.
     */
    void zero() noexcept
    {
        x = 0.0;
        y = 0.0;
        z = 0.0;
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
        double alat,
        const std::array<double, 9>& basis,
        const Vec3& a,
        const Vec3& b) noexcept
    {
        const double dx = b.x - a.x;
        const double dy = b.y - a.y;
        const double dz = b.z - a.z;

        const auto M = [&](int r, int c) noexcept { return basis[r * 3 + c]; };

        const double s0 = M(0,0)*dx + M(0,1)*dy + M(0,2)*dz;
        const double s1 = M(1,0)*dx + M(1,1)*dy + M(1,2)*dz;
        const double s2 = M(2,0)*dx + M(2,1)*dy + M(2,2)*dz;

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
    friend inline Vec3 operator*(const Vec3& a, double s)
    {
        return Vec3(a.x * s, a.y * s, a.z * s);
    }

    /** @brief Scalar multiplication */
    friend inline Vec3 operator*(double s, const Vec3& a)
    {
        return a * s;
    }

    /** @brief Matrix-vector multiplication (C-style 3x3 matrix) */
    friend inline Vec3 operator*(const double (&m)[3][3], const Vec3& v)
    {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        };
    }

    /** @brief Matrix-vector multiplication (std::array basis) */
    friend inline Vec3 operator*(const std::array<double, 9>& basis, const Vec3& v)
    {
        return {
            basis[0]*v.x + basis[1]*v.y + basis[2]*v.z,
            basis[3]*v.x + basis[4]*v.y + basis[5]*v.z,
            basis[6]*v.x + basis[7]*v.y + basis[8]*v.z
        };
    }

    /** @brief Scalar division */
    friend inline Vec3 operator/(const Vec3& a, double s)
    {
        if (s == 0.0) THROW_INVALID_ARGUMENT("Divide by zero error.");
        return Vec3(a.x / s, a.y / s, a.z / s);
    }
};

} // namespace Maths

#endif // VEC3_HPP
```

## A.2 OpenMP parallel loop example (canonical)
```cpp
void parallel_add(const std::vector<double>& numbers_a,
                  const std::vector<double>& numbers_b,
                  std::vector<double>& numbers_c)
{
    const auto n {std::size_t(numbers_a.size())};

    #pragma omp parallel for simd schedule(static)
    for (auto i = std::size_t(0); i < n; i++)
    {
        numbers_c[i] = numbers_a[i] + numbers_b[i];
    }
}
```

## A.3 Helper macro examples (canonical)
```cpp
#ifndef MACROS_HPP
#define MACROS_HPP

/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
/*********************************************************************************************************************************/

#define CLASS_SET_GET(Class, attribute)                                      \
    inline void set_##attribute(Class& attribute)                            \
    {                                                                        \
        _##attribute = std::move(attribute);                                 \
    }                                                                        \
                                                                             \
    Class& get_##attribute()                                                 \
    {                                                                        \
        return _##attribute;                                                 \
    }                                                                        \
                                                                             \
    const Class& get_##attribute() const                                     \
    {                                                                        \
        return _##attribute;                                                 \
    }                                                                        \
                                                                             \
    Class get_##attribute##_copy() const                                     \
    {                                                                        \
        return _##attribute;                                                 \
    }

#define DOUBLE_SET_GET(attribute)                                            \
    inline void set_##attribute(double attribute)                            \
    {                                                                        \
        _##attribute = attribute;                                            \
    }                                                                        \
                                                                             \
    double get_##attribute() const                                           \
    {                                                                        \
        return _##attribute;                                                 \
    }

#define SIZE_T_SET_GET(attribute)                                            \
    inline void set_##attribute(std::size_t attribute)                       \
    {                                                                        \
        _##attribute = attribute;                                            \
    }                                                                        \
                                                                             \
    std::size_t get_##attribute() const                                      \
    {                                                                        \
        return _##attribute;                                                 \
    }

#define STRING_SET_GET(attribute)                                            \
    inline void set_##attribute(std::string& attribute)                      \
    {                                                                        \
        _##attribute = std::move(attribute);                                 \
    }                                                                        \
                                                                             \
    std::string& get_##attribute()                                           \
    {                                                                        \
        return _##attribute;                                                 \
    }                                                                        \
                                                                             \
    const std::string& get_##attribute() const                               \
    {                                                                        \
        return _##attribute;                                                 \
    }                                                                        \
                                                                             \
    std::string get_##attribute##_copy() const                               \
    {                                                                        \
        return _##attribute;                                                 \
    }

#define ARRAY9_SET_GET(attribute)                                            \
    inline void set_##attribute(const std::array<double, 9>& attribute)      \
    {                                                                        \
        _##attribute = attribute;                                            \
    }                                                                        \
                                                                             \
    inline const std::array<double, 9>& get_##attribute() const              \
    {                                                                        \
        return _##attribute;                                                 \
    }                                                                        \
                                                                             \
    inline std::array<double, 9>& get_##attribute()                          \
    {                                                                        \
        return _##attribute;                                                 \
    }

#define SINGLETON(Class)                                                     \
    class Class##Once                                                        \
    {                                                                        \
    public:                                                                  \
        inline static Class& get() { static Class instance; return instance; }       \
        inline static const Class& get_ro() { static Class instance; return instance; } \
    private:                                                                 \
        Class##Once() {};                                                    \
        ~Class##Once() = default;                                            \
        Class##Once(const Class##Once&) = delete;                            \
        Class##Once& operator=(const Class##Once&) = delete;                 \
        Class##Once(Class##Once&&) = delete;                                 \
        Class##Once& operator=(const Class##Once&&) = delete;                \
    };

#define ITERATOR(Type, Attribute)                                            \
    std::vector<Type>::iterator begin() { return Attribute.begin(); }        \
    std::vector<Type>::iterator end()   { return Attribute.end(); }          \
    std::vector<Type>::const_iterator begin() const { return Attribute.begin(); } \
    std::vector<Type>::const_iterator end()   const { return Attribute.end(); } \
    std::vector<Type>::const_iterator cbegin() const { return Attribute.cbegin(); } \
    std::vector<Type>::const_iterator cend()   const { return Attribute.cend(); }

#endif
```

## A.4 Error macro examples (canonical)
```cpp
#ifndef ERROR_HPP
#define ERROR_HPP

#include <iostream>
#include <stdexcept>
#include <string>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define THROW_INVALID_ARGUMENT(msg) \
    throw std::invalid_argument( \
        std::string("\n[INVALID_ARGUMENT] \nError message:  ") + msg + \
        "\n  File: " + __FILE__ + \
        "\n  Line: " + TOSTRING(__LINE__) + \
        "\n  Function: " + __func__ \
    )

#define THROW_RUNTIME_ERROR(msg) \
    throw std::runtime_error( \
        std::string("\n[RUNTIME_ERROR] \nError message:  ") + msg + \
        "\n  File: " + __FILE__ + \
        "\n  Line: " + TOSTRING(__LINE__) + \
        "\n  Function: " + __func__ \
    )

#endif
```

## A.5 Struct example (canonical)
```cpp
#ifndef ATOM_HPP
#define ATOM_HPP

/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
/*********************************************************************************************************************************/

namespace SimpleMD
{

struct Atom
{

public:

    std::size_t atom_type {};
    std::size_t atom_id {};

    Maths::Vec3 position {};
    Maths::Vec3 force {};
    Maths::Vec3 velocity {};
    double mass {};
    double inv_mass {};
    Maths::Vec3 scratch {};

    Atom() {}
    Atom(const std::size_t id, const double x, const double y, const double z, const double mass) :
        atom_id(id), position(x, y, z), mass(std::max(mass, 1.0e-20)), inv_mass(1.0 / std::max(mass, 1.0e-20))
        {}

    void set_position(Maths::Vec3 position_in)
    {
        position = position_in;
        position.unit_cell_pbc();
    }

};

} // namespace SimpleMD

#endif
```

---

# Appendix: Additional Recommended Examples

## B.1 Fully-qualified namespace usage (no `using namespace`)
```cpp
#include <vector>

namespace SimpleMD
{

void example()
{
    auto values {std::vector<double>{1.0, 2.0, 3.0}};
    auto n {std::size_t(values.size())};
    (void)n;
}

} // namespace SimpleMD
```

## B.2 `enum class` + `switch` over long if/else
```cpp
namespace SimpleMD
{

enum class IntegratorKind
{
    VelocityVerlet,
    Langevin
};

void step(IntegratorKind kind)
{
    switch (kind)
    {
        case IntegratorKind::VelocityVerlet:
        {
            // ...
            break;
        }
        case IntegratorKind::Langevin:
        {
            // ...
            break;
        }
    }
}

} // namespace SimpleMD
```

## B.3 `std::variant` for modes (avoid polymorphism by default)
```cpp
#include <variant>

namespace SimpleMD
{

struct LennardJones
{
    double epsilon {};
    double sigma {};
};

struct Buckingham
{
    double A {};
    double B {};
    double C {};
};

using PairPotential = std::variant<LennardJones, Buckingham>;

double cutoff_radius(const PairPotential& pot)
{
    return std::visit(
        [](const auto& p) -> double
        {
            // Example: placeholder rule
            (void)p;
            return 2.5;
        },
        pot
    );
}

} // namespace SimpleMD
```

## B.4 Smart pointer ownership (no raw owning pointers)
```cpp
#include <memory>

namespace SimpleMD
{

class System
{
public:
    System() = default;

private:
    std::unique_ptr<int> data {}; // placeholder owned resource
};

} // namespace SimpleMD
```

## B.5 Non-owning views: prefer `std::span`
```cpp
#include <span>
#include <vector>

namespace SimpleMD
{

double sum(std::span<const double> values) noexcept
{
    auto total {0.0};
    for (auto v : values)
    {
        total += v;
    }
    return total;
}

void example(const std::vector<double>& v)
{
    auto total {sum(std::span<const double>(v.data(), v.size()))};
    (void)total;
}

} // namespace SimpleMD
```

## B.6 `std::format` for formatting output (avoid iostream formatting gymnastics)
```cpp
#include <format>
#include <string>

namespace SimpleMD
{

std::string format_step(std::size_t step_index, double time)
{
    return std::format("Step {} at t = {:.6f}\n", step_index, time);
}

} // namespace SimpleMD
```

## B.7 Precondition checking at API boundary (avoid exceptions in hot loops)
```cpp
#include <vector>

void checked_parallel_add(const std::vector<double>& numbers_a,
                          const std::vector<double>& numbers_b,
                          std::vector<double>& numbers_c)
{
    if (numbers_a.size() != numbers_b.size())
    {
        THROW_INVALID_ARGUMENT("Input vectors must have the same length.");
    }
    if (numbers_c.size() != numbers_a.size())
    {
        THROW_INVALID_ARGUMENT("Output vector must be pre-sized to match inputs.");
    }

    const auto n {std::size_t(numbers_a.size())};

    #pragma omp parallel for simd schedule(static)
    for (auto i = std::size_t(0); i < n; i++)
    {
        numbers_c[i] = numbers_a[i] + numbers_b[i];
    }
}
```
