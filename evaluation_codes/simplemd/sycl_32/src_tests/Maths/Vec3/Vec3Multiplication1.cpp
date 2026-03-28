#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ---- Multiply by zero ----
TEST(Vec3MultiplyScalar, MultiplyByZeroX)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = a * 0.0;
    const double found {r.x};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalar, MultiplyByZeroY)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = a * 0.0;
    const double found {r.y};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalar, MultiplyByZeroZ)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = a * 0.0;
    const double found {r.z};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Multiply by positive scalar ----
TEST(Vec3MultiplyScalar, MultiplyByPositiveX)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = a * 2.0;
    const double found {r.x};
    const double expected {2.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalar, MultiplyByPositiveY)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = a * 2.0;
    const double found {r.y};
    const double expected {-4.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalar, MultiplyByPositiveZ)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = a * 2.0;
    const double found {r.z};
    const double expected {6.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Multiply by negative scalar ----
TEST(Vec3MultiplyScalar, MultiplyByNegativeX)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = a * -1.5;
    const double found {r.x};
    const double expected {-1.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalar, MultiplyByNegativeY)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = a * -1.5;
    const double found {r.y};
    const double expected {3.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalar, MultiplyByNegativeZ)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = a * -1.5;
    const double found {r.z};
    const double expected {-4.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Multiply by large scalar ----
TEST(Vec3MultiplyScalar, MultiplyByLargeScalarX)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 r = a * A;
    const double found {r.x};
    const double expected {1.0e12};
    const double tolerance {1.0e3}; // relative ~1e-9
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalar, MultiplyByLargeScalarY)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 r = a * A;
    const double found {r.y};
    const double expected {2.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalar, MultiplyByLargeScalarZ)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 r = a * A;
    const double found {r.z};
    const double expected {3.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}
