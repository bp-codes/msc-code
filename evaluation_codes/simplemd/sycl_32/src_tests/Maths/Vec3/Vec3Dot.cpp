/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



// ---- Dot with zero vector ----
TEST(Vec3Dot, WithZeroVectorIsZero)
{
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(0.0, 0.0, 0.0);
    const double found {a.dot(b)};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Dot of vector with itself (squared length) ----
TEST(Vec3Dot, WithItselfEqualsLengthSquared)
{
    const Maths::Vec3 a(3.0, 4.0, 12.0);
    const double found {a.dot(a)};
    const double expected {169.0}; // 9 + 16 + 144
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Orthogonal vectors ----
TEST(Vec3Dot, OrthogonalVectorsGiveZero)
{
    const Maths::Vec3 a(1.0, 0.0, 0.0);
    const Maths::Vec3 b(0.0, 1.0, 0.0);
    const double found {a.dot(b)};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Parallel unit vectors ----
TEST(Vec3Dot, ParallelUnitVectorsGiveOne)
{
    const Maths::Vec3 a(0.0, 0.0, 1.0);
    const Maths::Vec3 b(0.0, 0.0, 1.0);
    const double found {a.dot(b)};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Opposite unit vectors ----
TEST(Vec3Dot, OppositeUnitVectorsGiveMinusOne)
{
    const Maths::Vec3 a(1.0, 0.0, 0.0);
    const Maths::Vec3 b(-1.0, 0.0, 0.0);
    const double found {a.dot(b)};
    const double expected {-1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Dot, MixedPositiveAndNegativeComponents)
{
    const Maths::Vec3 a(1.5, -2.0, 3.0);
    const Maths::Vec3 b(-4.0, 0.5, 2.0);
    const double found {a.dot(b)};
    const double expected {-1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}
