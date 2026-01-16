#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ---- Multiply by zero ----
TEST(Vec3MultiplyScalarComm, MultiplyByZeroX)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = 0.0 * a;
    const double found {r.x};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalarComm, MultiplyByZeroY)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = 0.0 * a;
    const double found {r.y};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalarComm, MultiplyByZeroZ)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = 0.0 * a;
    const double found {r.z};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Multiply by positive scalar ----
TEST(Vec3MultiplyScalarComm, MultiplyByPositiveX)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = 2.0 * a;
    const double found {r.x};
    const double expected {2.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalarComm, MultiplyByPositiveY)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = 2.0 * a;
    const double found {r.y};
    const double expected {-4.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalarComm, MultiplyByPositiveZ)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = 2.0 * a;
    const double found {r.z};
    const double expected {6.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Multiply by negative scalar ----
TEST(Vec3MultiplyScalarComm, MultiplyByNegativeX)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = -1.5 * a;
    const double found {r.x};
    const double expected {-1.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalarComm, MultiplyByNegativeY)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = -1.5 * a;
    const double found {r.y};
    const double expected {3.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalarComm, MultiplyByNegativeZ)
{
    const Maths::Vec3 a(1.0, -2.0, 3.0);
    const Maths::Vec3 r = -1.5 * a;
    const double found {r.z};
    const double expected {-4.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Multiply by large scalar ----
TEST(Vec3MultiplyScalarComm, MultiplyByLargeScalarX)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 r = A * a;
    const double found {r.x};
    const double expected {1.0e12};
    const double tolerance {1.0e3}; // relative ~1e-9
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalarComm, MultiplyByLargeScalarY)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 r = A * a;
    const double found {r.y};
    const double expected {2.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MultiplyScalarComm, MultiplyByLargeScalarZ)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 r = A * a;
    const double found {r.z};
    const double expected {3.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}
