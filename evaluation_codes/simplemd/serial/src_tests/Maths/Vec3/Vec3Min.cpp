#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ======================= Vec3::min Test Suite =======================

// ---- Simple positive values ----
TEST(Vec3Min, SimplePositiveX)
{
    const Maths::Vec3 a(1.0, 5.0, 9.0);
    const Maths::Vec3 b(2.0, 4.0, 8.0);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.x};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Min, SimplePositiveY)
{
    const Maths::Vec3 a(1.0, 5.0, 9.0);
    const Maths::Vec3 b(2.0, 4.0, 8.0);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.y};
    const double expected {4.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Min, SimplePositiveZ)
{
    const Maths::Vec3 a(1.0, 5.0, 9.0);
    const Maths::Vec3 b(2.0, 4.0, 8.0);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.z};
    const double expected {8.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- With negative numbers ----
TEST(Vec3Min, WorksWithNegativesX)
{
    const Maths::Vec3 a(-3.0, 2.0, -5.0);
    const Maths::Vec3 b(-1.0, -4.0, 6.0);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.x};
    const double expected {-3.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Min, WorksWithNegativesY)
{
    const Maths::Vec3 a(-3.0, 2.0, -5.0);
    const Maths::Vec3 b(-1.0, -4.0, 6.0);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.y};
    const double expected {-4.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Min, WorksWithNegativesZ)
{
    const Maths::Vec3 a(-3.0, 2.0, -5.0);
    const Maths::Vec3 b(-1.0, -4.0, 6.0);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.z};
    const double expected {-5.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Identical vectors ----
TEST(Vec3Min, IdenticalVectorsX)
{
    const Maths::Vec3 a(7.7, -8.8, 9.9);
    const Maths::Vec3 b(7.7, -8.8, 9.9);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.x};
    const double expected {7.7};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Min, IdenticalVectorsY)
{
    const Maths::Vec3 a(7.7, -8.8, 9.9);
    const Maths::Vec3 b(7.7, -8.8, 9.9);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.y};
    const double expected {-8.8};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Min, IdenticalVectorsZ)
{
    const Maths::Vec3 a(7.7, -8.8, 9.9);
    const Maths::Vec3 b(7.7, -8.8, 9.9);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.z};
    const double expected {9.9};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitudes ----
TEST(Vec3Min, WorksWithLargeMagnitudesX)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(A, -A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.x};
    const double expected {-1.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Min, WorksWithLargeMagnitudesY)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(A, -A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.y};
    const double expected {-1.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Min, WorksWithLargeMagnitudesZ)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(A, -A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = Maths::Vec3::min(a, b);
    const double found {r.z};
    const double expected {-1.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}
