/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



TEST(Vec3LengthSquared, ZeroVectorIsZero)
{
    const Maths::Vec3 v(0.0, 0.0, 0.0);
    const double found {v.length_squared()};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3LengthSquared, UnitXIsOne)
{
    const Maths::Vec3 v(1.0, 0.0, 0.0);
    const double found {v.length_squared()};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3LengthSquared, UnitYIsOne)
{
    const Maths::Vec3 v(0.0, 1.0, 0.0);
    const double found {v.length_squared()};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3LengthSquared, UnitZIsOne)
{
    const Maths::Vec3 v(0.0, 0.0, 1.0);
    const double found {v.length_squared()};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Pythagorean examples ----
TEST(Vec3LengthSquared, ThreeFourZeroGivesTwentyFive)
{
    const Maths::Vec3 v(3.0, 4.0, 0.0);
    const double found {v.length_squared()};
    const double expected {25.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3LengthSquared, ThreeFourTwelveGivesOneSixtyNine)
{
    const Maths::Vec3 v(3.0, 4.0, 12.0);
    const double found {v.length_squared()};
    const double expected {169.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Negative values should square positive ----
TEST(Vec3LengthSquared, NegativesStillPositive)
{
    const Maths::Vec3 v(-3.0, -4.0, 0.0);
    const double found {v.length_squared()};
    const double expected {25.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Decimal values ----
TEST(Vec3LengthSquared, DecimalComponents)
{
    const Maths::Vec3 v(1.5, -2.5, 2.0);
    const double found {v.length_squared()};
    const double expected {12.5}; // 1.5^2 + (-2.5)^2 + 2^2 = 2.25 + 6.25 + 4.0
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitude values ----
TEST(Vec3LengthSquared, WorksWithLargeMagnitudes)
{
    const double A = 1.0e150;
    const Maths::Vec3 v(A, A, A);
    const double found {v.length_squared()};
    const double expected {3.0e300}; // 3 * (1e150^2)
    const double tolerance {1.0e291}; // relative ~1e-9
    EXPECT_NEAR(found, expected, tolerance);
}

