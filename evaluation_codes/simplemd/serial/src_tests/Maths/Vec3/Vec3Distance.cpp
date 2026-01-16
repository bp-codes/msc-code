/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



// ---- Zero distance ----
TEST(Vec3Distance, SameVectorIsZero)
{
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(1.0, 2.0, 3.0);
    const double found {a.distance(b)};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Unit distance along X ----
TEST(Vec3Distance, UnitApartOnX)
{
    const Maths::Vec3 a(0.0, 0.0, 0.0);
    const Maths::Vec3 b(1.0, 0.0, 0.0);
    const double found {a.distance(b)};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Unit distance along Y ----
TEST(Vec3Distance, UnitApartOnY)
{
    const Maths::Vec3 a(0.0, 0.0, 0.0);
    const Maths::Vec3 b(0.0, 1.0, 0.0);
    const double found {a.distance(b)};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Unit distance along Z ----
TEST(Vec3Distance, UnitApartOnZ)
{
    const Maths::Vec3 a(0.0, 0.0, 0.0);
    const Maths::Vec3 b(0.0, 0.0, -1.0);
    const double found {a.distance(b)};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Pythagorean example (3-4-5) ----
TEST(Vec3Distance, ThreeFourZeroGivesFive)
{
    const Maths::Vec3 a(0.0, 0.0, 0.0);
    const Maths::Vec3 b(3.0, 4.0, 0.0);
    const double found {a.distance(b)};
    const double expected {5.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Opposite corners of a cube (1,1,1) vs (0,0,0) ----
TEST(Vec3Distance, CubeDiagonal)
{
    const Maths::Vec3 a(0.0, 0.0, 0.0);
    const Maths::Vec3 b(1.0, 1.0, 1.0);
    const double found {a.distance(b)};
    const double expected {1.732050807568877}; // sqrt(3)
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Mixed positive and negative components ----
TEST(Vec3Distance, MixedSigns)
{
    const Maths::Vec3 a(1.5, -2.0, 3.0);
    const Maths::Vec3 b(-4.0, 0.5, 2.0);
    const double found {a.distance(b)};
    const double expected {6.12372435695795}; // precomputed
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitude vectors ----
TEST(Vec3Distance, WorksWithLargeMagnitudes)
{
    const double A = 1.0e150;
    const Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, -A, -A);
    const double found {a.distance(b)};
    const double expected {3.464101615137754e150}; // 2A*sqrt(3)
    const double tolerance {1.0e141}; // relative ~1e-9
    EXPECT_NEAR(found, expected, tolerance);
}
