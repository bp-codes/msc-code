#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ======================= Vec3::operator*= Test Suite =======================

// ---- Multiply by zero ----
TEST(Vec3TimesEquals, MultiplyByZeroX)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a *= 0.0;
    const double found {a.x};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3TimesEquals, MultiplyByZeroY)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a *= 0.0;
    const double found {a.y};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3TimesEquals, MultiplyByZeroZ)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a *= 0.0;
    const double found {a.z};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Multiply by positive scalar ----
TEST(Vec3TimesEquals, MultiplyByPositiveX)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a *= 2.0;
    const double found {a.x};
    const double expected {2.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3TimesEquals, MultiplyByPositiveY)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a *= 2.0;
    const double found {a.y};
    const double expected {-4.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3TimesEquals, MultiplyByPositiveZ)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a *= 2.0;
    const double found {a.z};
    const double expected {6.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Multiply by negative scalar ----
TEST(Vec3TimesEquals, MultiplyByNegativeX)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a *= -1.5;
    const double found {a.x};
    const double expected {-1.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3TimesEquals, MultiplyByNegativeY)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a *= -1.5;
    const double found {a.y};
    const double expected {3.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3TimesEquals, MultiplyByNegativeZ)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a *= -1.5;
    const double found {a.z};
    const double expected {-4.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitude scalar ----
TEST(Vec3TimesEquals, MultiplyByLargeScalarX)
{
    const double S = 1.0e12;
    Maths::Vec3 a(1.0, 2.0, 3.0);
    a *= S;
    const double found {a.x};
    const double expected {1.0e12};
    const double tolerance {1.0e3}; // ~1e-9 relative
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3TimesEquals, MultiplyByLargeScalarY)
{
    const double S = 1.0e12;
    Maths::Vec3 a(1.0, 2.0, 3.0);
    a *= S;
    const double found {a.y};
    const double expected {2.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3TimesEquals, MultiplyByLargeScalarZ)
{
    const double S = 1.0e12;
    Maths::Vec3 a(1.0, 2.0, 3.0);
    a *= S;
    const double found {a.z};
    const double expected {3.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Returns reference to self ----
TEST(Vec3TimesEquals, ReturnsReferenceToSelf)
{
    Maths::Vec3 a(1.0, 2.0, 3.0);
    Maths::Vec3& ref = (a *= 2.0);
    EXPECT_EQ(&ref, &a);
}
