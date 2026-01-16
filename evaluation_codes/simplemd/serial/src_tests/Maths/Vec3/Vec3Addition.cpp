#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ---- Simple addition ----
TEST(Vec3Addition, AddsPositiveComponentsX)
{
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(4.0, 5.0, 6.0);
    const Maths::Vec3 r = a + b;
    const double found {r.x};
    const double expected {5.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Addition, AddsPositiveComponentsY)
{
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(4.0, 5.0, 6.0);
    const Maths::Vec3 r = a + b;
    const double found {r.y};
    const double expected {7.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Addition, AddsPositiveComponentsZ)
{
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(4.0, 5.0, 6.0);
    const Maths::Vec3 r = a + b;
    const double found {r.z};
    const double expected {9.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- With negative components ----
TEST(Vec3Addition, AddsWithNegativesX)
{
    const Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    const Maths::Vec3 r = a + b;
    const double found {r.x};
    const double expected {3.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Addition, AddsWithNegativesY)
{
    const Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    const Maths::Vec3 r = a + b;
    const double found {r.y};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Addition, AddsWithNegativesZ)
{
    const Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    const Maths::Vec3 r = a + b;
    const double found {r.z};
    const double expected {-2.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitudes ----
TEST(Vec3Addition, WorksWithLargeMagnitudesX)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = a + b;
    const double found {r.x};
    const double expected {0.0};
    const double tolerance {1.0e-3}; // allow looser tolerance for big values
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Addition, WorksWithLargeMagnitudesY)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = a + b;
    const double found {r.y};
    const double expected {2.0e12};
    const double tolerance {1.0e3}; // relative ~1e-9
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Addition, WorksWithLargeMagnitudesZ)
{
    const double A = 1.0e12;
    const Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = a + b;
    const double found {r.z};
    const double expected {0.0};
    const double tolerance {1.0e-3};
    EXPECT_NEAR(found, expected, tolerance);
}
