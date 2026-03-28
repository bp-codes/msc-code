#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ---- Divide by nonzero scalar ----
TEST(Vec3DivideScalar, DivideByPositiveX)
{
    const Maths::Vec3 a(2.0, -4.0, 6.0);
    const Maths::Vec3 r = a / 2.0;
    const double found {r.x};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideScalar, DivideByPositiveY)
{
    const Maths::Vec3 a(2.0, -4.0, 6.0);
    const Maths::Vec3 r = a / 2.0;
    const double found {r.y};
    const double expected {-2.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideScalar, DivideByPositiveZ)
{
    const Maths::Vec3 a(2.0, -4.0, 6.0);
    const Maths::Vec3 r = a / 2.0;
    const double found {r.z};
    const double expected {3.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Divide by negative scalar ----
TEST(Vec3DivideScalar, DivideByNegativeX)
{
    const Maths::Vec3 a(3.0, -6.0, 9.0);
    const Maths::Vec3 r = a / -3.0;
    const double found {r.x};
    const double expected {-1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideScalar, DivideByNegativeY)
{
    const Maths::Vec3 a(3.0, -6.0, 9.0);
    const Maths::Vec3 r = a / -3.0;
    const double found {r.y};
    const double expected {2.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideScalar, DivideByNegativeZ)
{
    const Maths::Vec3 a(3.0, -6.0, 9.0);
    const Maths::Vec3 r = a / -3.0;
    const double found {r.z};
    const double expected {-3.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitude division ----
TEST(Vec3DivideScalar, DivideLargeValues)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(A, -A, A);
    const Maths::Vec3 r = a / 1.0e6;
    const double found {r.x};
    const double expected {1.0e6};
    const double tolerance {1.0e-3}; // relative ~1e-9
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Divide by zero must throw ----
TEST(Vec3DivideScalar, DivideByZeroThrows)
{
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    EXPECT_THROW({
        const Maths::Vec3 r = a / 0.0;
        (void)r;
    }, std::invalid_argument);
}
