#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ======================= Vec3::max Test Suite =======================

// ---- Simple positive values ----
TEST(Vec3Max, SimplePositiveX)
{
    const Maths::Vec3 a(1.0, 5.0, 9.0);
    const Maths::Vec3 b(2.0, 4.0, 8.0);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.x};
    const float expected {2.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Max, SimplePositiveY)
{
    const Maths::Vec3 a(1.0, 5.0, 9.0);
    const Maths::Vec3 b(2.0, 4.0, 8.0);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.y};
    const float expected {5.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Max, SimplePositiveZ)
{
    const Maths::Vec3 a(1.0, 5.0, 9.0);
    const Maths::Vec3 b(2.0, 4.0, 8.0);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.z};
    const float expected {9.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- With negative numbers ----
TEST(Vec3Max, WorksWithNegativesX)
{
    const Maths::Vec3 a(-3.0, 2.0, -5.0);
    const Maths::Vec3 b(-1.0, -4.0, 6.0);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.x};
    const float expected {-1.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Max, WorksWithNegativesY)
{
    const Maths::Vec3 a(-3.0, 2.0, -5.0);
    const Maths::Vec3 b(-1.0, -4.0, 6.0);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.y};
    const float expected {2.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Max, WorksWithNegativesZ)
{
    const Maths::Vec3 a(-3.0, 2.0, -5.0);
    const Maths::Vec3 b(-1.0, -4.0, 6.0);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.z};
    const float expected {6.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Identical vectors ----
TEST(Vec3Max, IdenticalVectorsX)
{
    const Maths::Vec3 a(7.7, -8.8, 9.9);
    const Maths::Vec3 b(7.7, -8.8, 9.9);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.x};
    const float expected {7.7};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Max, IdenticalVectorsY)
{
    const Maths::Vec3 a(7.7, -8.8, 9.9);
    const Maths::Vec3 b(7.7, -8.8, 9.9);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.y};
    const float expected {-8.8};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Max, IdenticalVectorsZ)
{
    const Maths::Vec3 a(7.7, -8.8, 9.9);
    const Maths::Vec3 b(7.7, -8.8, 9.9);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.z};
    const float expected {9.9};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitudes ----
TEST(Vec3Max, WorksWithLargeMagnitudesX)
{
    const float A = 1.0e12;
    const Maths::Vec3 a(A, -A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.x};
    const float expected {1.0e12};
    const float tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Max, WorksWithLargeMagnitudesY)
{
    const float A = 1.0e12;
    const Maths::Vec3 a(A, -A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.y};
    const float expected {1.0e12};
    const float tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Max, WorksWithLargeMagnitudesZ)
{
    const float A = 1.0e12;
    const Maths::Vec3 a(A, -A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = Maths::Vec3::max(a, b);
    const float found {r.z};
    const float expected {1.0e12};
    const float tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}
