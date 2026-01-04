# C++23 Coding Standard (Scientific / HPC)

## Purpose and rule keywords

This coding standard promotes **best practices** for scientific/HPC software: a consistent look and feel, maintainable APIs, predictable performance, and **memory safety** through clear ownership and lifetime rules. It is also intended to make code reviews faster and more objective by defining explicit, testable expectations.

### Keyword meanings

The following keywords indicate requirement levels in this document:

- **MUST / MUST NOT**  
  Mandatory requirement. Code that violates a MUST requires an approved exception.

- **SHALL / SHALL NOT**  
  Mandatory requirement, equivalent to **MUST / MUST NOT**. It indicates a strict obligation with no discretion. Any deviation requires an approved exception.

- **SHOULD / SHOULD NOT**  
  Strong recommendation. Deviations are allowed only with a clear, documented justification (e.g., measurable performance benefit or required interoperability).

- **MAY**  
  Optional guidance. Use when it improves clarity, safety, or performance.

---

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
- **MUST** name include guards as `UPPERCASE_FILENAME_HPP` (or `_IXX` if needed).
  Example:
  ```cpp
  #ifndef VEC3_HPP
  #define VEC3_HPP
  // ...
  #endif
  ```
- **SHOULD** place a project-local include banner/separator consistently. 

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
- **SHOULD** separate functions with 3 blank lines.
- **SHOULD** have no more than one blank line within a function.

---

## 4) Naming conventions
- **MUST** use:
  - variables: `snake_case` (`my_integer`)
  - types/classes/structs: `PascalCase` (`MyClass`)
  - macro-like parameters: `UPPERCASE` (`MYPARAMETER`)
  - constants: `UPPERCASE_WITH_UNDERSCORES` (`MY_CONST_INTEGER`)
- **MUST** name **private data members** with a leading underscore (e.g., `_count`, `_atoms`).  
  *Rationale:* consistent backing-field naming (matches project macros) and makes ownership/encapsulation obvious.
- **MAY** omit the leading underscore in **pure data structs** that intentionally expose public fields (e.g., POD-like GPU-portable structs).  
  *Rationale:* keeps data layouts simple when encapsulation is not the goal.
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
- **SHOULD** keep header-only utilities `inline`/`constexpr` as appropriate.

---

## 6) Namespaces and symbol visibility
- **MUST** use namespaces for libraries and logical subsystems (e.g., `Maths`, `SimpleMD`).  
  *Rationale:* prevents global pollution, improves organization.
- **MUST NOT** use `using namespace ...` at file scope.  
  *Rationale:* avoids symbol collisions and unclear provenance.
- **MUST NOT** use `using std::foo;` at file scope unless explicitly allowed by project policy (default: not allowed).  
  *Rationale:* same as above; “full naming only”.
- **SHOULD** close namespaces with a trailing comment. 
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
- **MUST** avoid C-style casts.  
  **Prefer:** `static_cast`, `const_cast`, `reinterpret_cast` (rare), as appropriate.  
  *Rationale:* explicit intent and safer reviews.
- **SHOULD** keep conversions explicit at type boundaries (e.g., `double` ↔ `std::size_t`).  
  Example:
  ```cpp
  auto n {std::size_t(numbers_a.size())};
  auto i {std::size_t(0)};
  ```
- **SHOULD** mark value-returning helpers as `[[nodiscard]]` when ignoring the result is likely a bug.

---

## 8) Const-correctness and `[[nodiscard]]`
- **MUST** make parameters `const` when they are inputs only.  
  *Rationale:* communicates intent and enables optimization.
- **MUST** prefer `const T&` for non-trivial input types (e.g., vectors, structs) unless copying is intended.  
  *Rationale:* avoids needless copies.
- **SHOULD** mark:
  - pure computations returning a value as `[[nodiscard]]`
  - functions that create derived values (norms, distances, formatted strings, etc.) as `[[nodiscard]]`  
  *Rationale:* helps catch “computed but unused” bugs in numerical code.
- **MAY** omit `[[nodiscard]]` for trivial accessors or when discarding is clearly normal.

---

## 9) Ownership, lifetime, and pointers
- **MUST** avoid raw owning pointers. Use `std::unique_ptr` / `std::shared_ptr` for ownership only.  
  *Rationale:* prevents leaks and unclear ownership.
- **MAY** use raw pointers only as **non-owning observers** when references/spans/iterators are unsuitable.  
  *Rationale:* allows interop and low-level APIs while preserving ownership clarity.
- **SHOULD** prefer references for required non-null inputs; prefer `std::span` for array-like views.  
  *Rationale:* explicit intent, easier optimization.
- **SHOULD** avoid unnecessary heap allocation in hot paths.  
  *Rationale:* HPC performance.
- **MUST** use `std::variant` over polymorphism by default; use polymorphism only when runtime extensibility or type erasure is clearly beneficial.  
  *Rationale:* variant enables value semantics and avoids vtables.

---

## 10) Control flow guidelines (if/switch/enums)
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

## 11) STL usage and algorithms/iterators
- **MUST** use the STL where available instead of rewriting common utilities.  
  *Rationale:* correctness, portability, maintainability.
- **SHOULD** prefer standard algorithms and iterators where they improve clarity.  
  *Rationale:* expresses intent and reduces bugs.
- **MAY** write explicit loops in performance-critical sections when it improves vectorization or clarity.  
  *Rationale:* HPC kernels often benefit from explicit loops.
- **SHOULD** use `std::clamp`, `std::min`, `std::max`, etc. (as shown). 

---

## 12) Performance and allocations (hot path policy)
- **MUST** assume inner loops and OpenMP regions are performance-critical unless stated otherwise.  
  *Rationale:* HPC default.
- **MUST** avoid hidden allocations in hot paths:
  - no `std::string` construction/concatenation in kernels
  - no `std::format` / iostream formatting in kernels
  - no repeated `push_back` without `reserve` when sizes are known
- **SHOULD** pre-size outputs and validate sizes before entering parallel regions.  
  *Rationale:* avoids races and per-iteration overhead.
- **SHOULD** prefer `std::span`/iterators for views to avoid copying.  
  *Rationale:* keeps data movement explicit.

---

## 13) Error handling and contracts (assertions/exceptions)
- **MUST** use project error macros for throwing exceptions with context:
  - `THROW_INVALID_ARGUMENT(msg)`
  - `THROW_RUNTIME_ERROR(msg)`  
  *Rationale:* consistent, provides file/line/function context (examples).
- **SHOULD** throw `std::invalid_argument` for precondition violations at API boundaries.  
  *Rationale:* predictable error taxonomy.
- **MUST** avoid exceptions inside hot inner loops unless unavoidable; validate inputs before entering kernels.  
  *Rationale:* performance and control flow predictability.
- **SHOULD** mark non-throwing functions `noexcept` where correct.

---

## 14) Parallelism (OpenMP)
- **MUST** use OpenMP for parallel sections where appropriate.  
  *Rationale:* canonical rule and matches example usage.
- **SHOULD** keep OpenMP pragmas tightly scoped to the smallest loop/region needed.  
  *Rationale:* avoids unintended sharing and makes correctness reviewable.
- **MUST** ensure container sizes match and outputs are sized before the parallel region.  
  *Rationale:* avoids races/out-of-bounds in parallel loops.
- **SHOULD** prefer `schedule(static)` for uniform work in tight numerical loops unless profiling shows otherwise. 
- **MAY** use `simd` when it improves vectorization and correctness is maintained.
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

## 15) Documentation (Doxygen)
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

## 16) Macro policy (minimal macros)
- **MUST** keep macro usage to a minimum. Prefer:
  - `inline` functions
  - templates
  - `constexpr` variables/functions
  - `enum class`
  *Rationale:* macros reduce type safety and tooling support.
- **MUST** keep macros in dedicated headers (e.g., `Macros.hpp`, `Error.hpp`) and avoid spreading new macros across the codebase.  
  *Rationale:* limits blast radius.
- **MUST** name macros in `UPPERCASE_WITH_UNDERSCORES`.  
  *Rationale:* makes macros obvious.
- **MUST NOT** introduce macros that:
  - evaluate arguments multiple times (unless explicitly documented and safe)
  - hide control flow (e.g., early returns, loops) without strong justification
- **MAY** use the existing project macros (set/get helpers, singleton helper, iterator helper, throw helpers) when they reduce boilerplate and improve consistency.
- **SHOULD** prefer explicit code over macros when the macro obscures behavior or makes debugging harder.

---

## 17) Forbidden patterns and preferred alternatives
- **MUST NOT** use raw owning pointers.  
  **Prefer:** `std::unique_ptr`, `std::shared_ptr`.
- **MUST NOT** use `#pragma once`.  
  **Prefer:** include guards.
- **MUST NOT** use `using namespace ...` at file scope.  
  **Prefer:** fully qualified names.
- **MUST NOT** use C-style casts.  
  **Prefer:** `static_cast` (and other C++ casts where appropriate).
- **SHOULD NOT** use `std::endl` for normal output.  
  **Prefer:** `'
'`.
- **SHOULD** use `std::format` for formatted output (if available).  
  *Rationale* A clearer method for outputting formatted data.
- **SHOULD NOT** write long `if/else` ladders on discrete modes.  
  **Prefer:** `enum class` + `switch`.
- **SHOULD NOT** reimplement STL utilities.  
  **Prefer:** `<algorithm>`, `<numeric>`, `<ranges>` when appropriate.
- **SHOULD** avoid unnecessary polymorphism.  
  **Prefer:** `std::variant` and value semantics by default.

---

## 18) Reviewer checklist
- Naming matches conventions (snake_case vars, PascalCase types, UPPERCASE constants/macros).
- Private members start with `_` (unless a pure data struct with public fields).
- 4 spaces, Allman braces, no tabs.
- Include guards present; no `#pragma once`.
- No `using namespace` at file scope; namespaces used appropriately.
- Initialization uses `{}`; `std::size_t` used for sizes/indices.
- `auto` used where types are obvious; not used where it obscures meaning.
- No C-style casts; conversions are explicit via `static_cast` etc.
- No raw owning pointers; ownership is explicit via smart pointers.
- Hot paths contain no hidden allocations/formatting/logging.
- STL used instead of custom reimplementations; algorithms/iterators used where sensible.
- `enum class` + `switch` used for modes; cases complete.
- Exceptions/macros used consistently (`THROW_INVALID_ARGUMENT`, `THROW_RUNTIME_ERROR`); no exceptions in hot loops.
- OpenMP pragmas are minimal-scope; loop correctness/race-safety checked.
- Public APIs have Doxygen docs (`@param`, `@return`, `@throws`, warnings where needed).
- Macros are minimal and only from the approved macro headers.

---

## Exceptions
- **Non-owning raw pointers** are **MAY**-allowed only for interoperability or low-level APIs where references/spans/iterators are impractical; ownership must remain elsewhere and be documented.
- **Explicit loops** may be preferred over STL algorithms in performance-critical kernels when it improves vectorization or clarity of the numerical operation.
- **Polymorphism** is permitted when runtime extension or plugin-style architecture is required and `std::variant` would be awkward or inefficient.

---

# Appendix: Canonical Examples (with full comments)

## A.1 Vec3 example (canonical, fully commented)

```cpp
#ifndef VEC3_HPP
#define VEC3_HPP

#include "_helper.hpp"

/**
 * @file Vec3.hpp
 * @brief 3D vector class with common linear algebra operations.
 *
 * Provides a lightweight 3-component vector type intended for numerical
 * and scientific computing. Supports arithmetic, norms, dot/cross products,
 * periodic boundary conditions, and basis transformations.
 */

// Maths
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
    // Data members
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

    /** @brief Default constructor. Initializes to (0,0,0). */
    constexpr Vec3() : x(0), y(0), z(0) {}

    /**
     * @brief Construct from components.
     * @param x x-component
     * @param y y-component
     * @param z z-component
     */
    constexpr Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    //#######################################
    // Operators and methods
    //#######################################

    /** @brief Unary minus. */
    [[nodiscard]]
    inline Vec3 operator-() const
    {
        return Vec3(-x, -y, -z);
    }

    /**
     * @brief Component access by index.
     * @param i Index (0=x, 1=y, 2=z)
     * @warning No bounds checking is performed.
     */
    inline double& operator[](int i)
    {
        return *((&x) + i);
    }

    /**
     * @brief Component access by index (const).
     * @param i Index (0=x, 1=y, 2=z)
     * @warning No bounds checking is performed.
     */
    inline const double& operator[](int i) const
    {
        return *((&x) + i);
    }

    /** @brief Euclidean norm. */
    [[nodiscard]]
    inline double length() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    /** @brief Squared Euclidean norm (avoids sqrt). */
    [[nodiscard]]
    inline double length_squared() const
    {
        return x * x + y * y + z * z;
    }

    /**
     * @brief Normalized copy of the vector.
     * @return Unit vector, or original vector if zero-length.
     */
    [[nodiscard]]
    inline Vec3 normalize() const
    {
        const double len = length();
        return len > 0.0 ? (*this) / len : *this;
    }

    /**
     * @brief Dot product.
     * @param other Other vector
     */
    [[nodiscard]]
    inline double dot(const Vec3& other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    /**
     * @brief Cross product.
     * @param other Other vector
     */
    [[nodiscard]]
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
     * @param other Other vector
     */
    [[nodiscard]]
    inline double distance(const Vec3& other) const
    {
        return (*this - other).length();
    }

    /**
     * @brief Clamp components to a given range.
     * @param minVal Minimum value
     * @param maxVal Maximum value
     */
    [[nodiscard]]
    inline Vec3 clamp(double minVal = 0.0, double maxVal = 1.0) const
    {
        return Vec3(
            std::clamp(x, minVal, maxVal),
            std::clamp(y, minVal, maxVal),
            std::clamp(z, minVal, maxVal)
        );
    }

    /**
     * @brief Apply periodic boundary conditions in unit cell [0,1).
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

    /** @brief In-place vector addition. */
    inline Vec3& operator+=(const Vec3& other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    /** @brief In-place vector subtraction. */
    inline Vec3& operator-=(const Vec3& other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    /** @brief In-place scalar multiplication. */
    inline Vec3& operator*=(double scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    /**
     * @brief In-place scalar division.
     * @throws std::invalid_argument if scalar is zero.
     */
    inline Vec3& operator/=(double scalar)
    {
        if (scalar == 0.0)
        {
            THROW_INVALID_ARGUMENT("Divide by zero error.");
        }
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    /** @brief Set all components to zero. */
    void zero() noexcept
    {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }

    //#######################################
    // Static utilities
    //#######################################

    /** @brief Component-wise minimum of two vectors. */
    [[nodiscard]]
    inline static Vec3 min(const Vec3& a, const Vec3& b) noexcept
    {
        return Vec3(
            std::min(a.x, b.x),
            std::min(a.y, b.y),
            std::min(a.z, b.z)
        );
    }

    /** @brief Component-wise maximum of two vectors. */
    [[nodiscard]]
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
     * @param alat Lattice constant
     * @param basis 3x3 basis matrix (row-major)
     * @param a First position
     * @param b Second position
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

        const auto M = [&](int r, int c) noexcept
        {
            return basis[static_cast<std::size_t>(r * 3 + c)];
        };

        const double s0 = M(0,0)*dx + M(0,1)*dy + M(0,2)*dz;
        const double s1 = M(1,0)*dx + M(1,1)*dy + M(1,2)*dz;
        const double s2 = M(2,0)*dx + M(2,1)*dy + M(2,2)*dz;

        return Vec3(alat * s0, alat * s1, alat * s2);
    }

    //#######################################
    // Friends
    //#######################################

    /** @brief Stream output operator. */
    friend inline std::ostream& operator<<(std::ostream& os, const Vec3& v)
    {
        return os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    }
};

} 
// Maths

#endif // VEC3_HPP
```



## A.2 OpenMP parallel loop example (commented)

```cpp
/**
 * @brief Element-wise addition of two vectors using OpenMP.
 *
 * @param numbers_a First input vector
 * @param numbers_b Second input vector
 * @param numbers_c Output vector (must be pre-sized)
 */
void parallel_add(const std::vector<double>& numbers_a,
                  const std::vector<double>& numbers_b,
                  std::vector<double>& numbers_c)
{
    const auto n {std::size_t(numbers_a.size())};

    // Parallel loop, attempt to use SIMD
    #pragma omp parallel for simd schedule(static)
    for (auto i = std::size_t(0); i < n; i++)
    {
        numbers_c[i] = numbers_a[i] + numbers_b[i];
    }
}
```



## A.3 Struct example

```cpp
#ifndef ATOM_HPP
#define ATOM_HPP

#include "_helper.hpp"
#include "_maths.hpp"

// SimpleMD
namespace SimpleMD
{

// Define a minimum mass to avoid divide by zero errors
inline constexpr double MIN_MASS {1.0e-20};

/**
 * @struct Atom
 * @brief Lightweight particle data structure.
 */
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

    /** @brief Default constructor. */
    Atom() = default;

    /**
     * @brief Construct atom with initial position and mass.
     */
    Atom(std::size_t id, double x, double y, double z, double mass_in) :
        atom_id(id),
        position(x, y, z)
    {
        mass = std::max(mass_in, SimpleMD::MIN_MASS);
        inv_mass = 1.0 / mass;
    }

    /**
     * @brief Set position and apply periodic boundary conditions.
     */
    void set_position(Maths::Vec3 position_in)
    {
        position = position_in;
        position.unit_cell_pbc();
    }
};

} 
// SimpleMD

#endif // ATOM_HPP
```



## A.4 Fully-qualified namespace usage (no `using namespace`)
```cpp
#include <vector>

/**
 * @namespace SimpleMD
 * @brief Core namespace for the SimpleMD molecular dynamics library.
 */
namespace SimpleMD
{

/**
 * @brief Example function demonstrating fully-qualified namespace usage.
 *
 * Shows use of standard library types without `using namespace`,
 * in accordance with project coding standards.
 */
void example()
{
    auto values {std::vector<double>{1.0, 2.0, 3.0}};
    auto n {std::size_t(values.size())};

    // Suppress unused-variable warning for demonstration purposes
    (void)n;
}

}
// SimpleMD
```



## A.5 `enum class` + `switch` over long if/else
```cpp

/**
 * @namespace SimpleMD
 * @brief Core namespace for the SimpleMD molecular dynamics library.
 */
namespace SimpleMD
{

/**
 * @enum IntegratorKind
 * @brief Enumeration of supported time-integration schemes.
 */
enum class IntegratorKind
{
    VelocityVerlet,
    Langevin
};

/**
 * @brief Perform a single integration step.
 *
 * Selects the integration algorithm based on the provided
 * integrator kind.
 *
 * @param kind Integration scheme to use.
 */
void step(IntegratorKind kind)
{
    switch (kind)
    {
        case IntegratorKind::VelocityVerlet:
        {
            // Velocity Verlet integration step
            break;
        }
        case IntegratorKind::Langevin:
        {
            // Langevin dynamics integration step
            break;
        }
    }
}

} 
// SimpleMD
```



## A.6 `std::variant` for modes (avoid polymorphism by default)
```cpp
#include <variant>

/**
 * @namespace SimpleMD
 * @brief Core namespace for the SimpleMD molecular dynamics library.
 */
namespace SimpleMD
{

/**
 * @struct LennardJones
 * @brief Lennard-Jones pair potential parameters.
 */
struct LennardJones
{
    double epsilon {};
    double sigma {};
};

/**
 * @struct Buckingham
 * @brief Buckingham pair potential parameters.
 */
struct Buckingham
{
    double A {};
    double B {};
    double C {};
};

/**
 * @brief Variant type representing supported pair potentials.
 *
 * Used in preference to runtime polymorphism.
 */
using PairPotential = std::variant<LennardJones, Buckingham>;

/**
 * @brief Return the cutoff radius for a given pair potential.
 *
 * @param pot Pair potential variant.
 * @return Cutoff radius.
 */
double cutoff_radius(const PairPotential& pot)
{
    return std::visit(
        [](const auto& p) -> double
        {
            // Placeholder rule; real implementation would
            // depend on potential parameters.
            (void)p;
            return 2.5;
        },
        pot
    );
}

}
// SimpleMD
```



## A.7 Smart pointer ownership (no raw owning pointers)
```cpp
#include <memory>

/**
 * @namespace SimpleMD
 * @brief Core namespace for the SimpleMD molecular dynamics library.
 */
namespace SimpleMD
{

/**
 * @class System
 * @brief Example system class demonstrating ownership via smart pointers.
 *
 * Owns its resources explicitly using `std::unique_ptr`.
 */
class System
{
public:
    /** @brief Default constructor. */
    System() = default;

private:
    /** @brief Owned resource (placeholder example). */
    std::unique_ptr<int> _data {};
};

}
// SimpleMD
```



## A.8 Non-owning views: prefer `std::span`
```cpp
#include <span>
#include <vector>

/**
 * @namespace SimpleMD
 * @brief Core namespace for the SimpleMD molecular dynamics library.
 */
namespace SimpleMD
{

/**
 * @brief Compute the sum of a sequence of values.
 *
 * Uses `std::span` to express a non-owning view of the data.
 *
 * @param values View of input values.
 * @return Sum of the values.
 */
double sum(std::span<const double> values) noexcept
{
    auto total {0.0};
    for (auto v : values)
    {
        total += v;
    }
    return total;
}

/**
 * @brief Example of passing a container as a span.
 *
 * @param v Input vector.
 */
void example(const std::vector<double>& v)
{
    auto total {sum(std::span<const double>(v.data(), v.size()))};
    (void)total;
}

}
// SimpleMD
```



## A.9 `std::format` for formatting output
```cpp
#include <format>
#include <string>

/**
 * @namespace SimpleMD
 * @brief Core namespace for the SimpleMD molecular dynamics library.
 */
namespace SimpleMD
{

/**
 * @brief Format a simulation step for output.
 *
 * @param step_index Simulation step index.
 * @param time Simulation time.
 * @return Formatted output string.
 */
std::string format_step(std::size_t step_index, double time)
{
    return std::format("Step {} at t = {:.6f}\n", step_index, time);
}

}
// SimpleMD
```



## A.10 Precondition checking
```cpp
#include <vector>

/**
 * @brief Add two vectors in parallel with precondition checking.
 *
 * Input validation is performed before entering the OpenMP region.
 *
 * @param numbers_a First input vector.
 * @param numbers_b Second input vector.
 * @param numbers_c Output vector (must be pre-sized).
 *
 * @throws std::invalid_argument if vector sizes do not match.
 */
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