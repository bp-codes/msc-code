#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ---- Basic addition ----
TEST(Vec3PlusEquals, AddsComponentsX)
{
    Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(4.0, 5.0, 6.0);
    a += b;
    const double found {a.x};
    const double expected {5.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3PlusEquals, AddsComponentsY)
{
    Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(4.0, 5.0, 6.0);
    a += b;
    const double found {a.y};
    const double expected {7.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3PlusEquals, AddsComponentsZ)
{
    Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(4.0, 5.0, 6.0);
    a += b;
    const double found {a.z};
    const double expected {9.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- With negatives ----
TEST(Vec3PlusEquals, WorksWithNegativeValuesX)
{
    Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    a += b;
    const double found {a.x};
    const double expected {3.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3PlusEquals, WorksWithNegativeValuesY)
{
    Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    a += b;
    const double found {a.y};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3PlusEquals, WorksWithNegativeValuesZ)
{
    Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    a += b;
    const double found {a.z};
    const double expected {-2.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitude values ----
TEST(Vec3PlusEquals, WorksWithLargeMagnitudesX)
{
    const double A = 1.0e12;
    Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    a += b;
    const double found {a.x};
    const double expected {0.0};
    const double tolerance {1.0e-3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3PlusEquals, WorksWithLargeMagnitudesY)
{
    const double A = 1.0e12;
    Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    a += b;
    const double found {a.y};
    const double expected {2.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3PlusEquals, WorksWithLargeMagnitudesZ)
{
    const double A = 1.0e12;
    Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    a += b;
    const double found {a.z};
    const double expected {0.0};
    const double tolerance {1.0e-3};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Returns reference to self ----
TEST(Vec3PlusEquals, ReturnsReferenceToSelf)
{
    Maths::Vec3 a(1.0, 2.0, 3.0);
    Maths::Vec3 b(4.0, 5.0, 6.0);
    Maths::Vec3& ref = (a += b);
    EXPECT_EQ(&ref, &a); // same address
}
