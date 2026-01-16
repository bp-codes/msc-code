/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



TEST(Vec3Length, ZeroVectorIsZero) 
{
    const Maths::Vec3 v(0.0, 0.0, 0.0);
    const double found {v.length()};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Length, UnitXHasLengthOne) 
{
    const Maths::Vec3 v(1.0, 0.0, 0.0);
    const double found {v.length()};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Length, UnitYHasLengthOne) 
{
    const Maths::Vec3 v(0.0, 1.0, 0.0);
    const double found {v.length()};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Length, UnitZHasLengthOne) 
{
    const Maths::Vec3 v(0.0, 0.0, 1.0);
    const double found {v.length()};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Length, ThreeFourZeroGivesFive) 
{
    const Maths::Vec3 v(3.0, 4.0, 0.0);
    const double found {v.length()};
    const double expected {5.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Length, ThreeFourTwelveGivesThirteen) 
{
    const Maths::Vec3 v(3.0, 4.0, 12.0);
    const double found {v.length()};
    const double expected {13.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Length, NegativesDontChangeLength) 
{
    const Maths::Vec3 v(-3.0, -4.0, 0.0);
    const double found {v.length()};
    const double expected {5.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Length, DecimalComponents) 
{
    const Maths::Vec3 v(1.5, -2.5, 2.0);
    const double found {v.length()};
    const double expected {3.535533906};  // precomputed
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Length, WorksWithLargeMagnitudes) 
{
    const double A = 1.0e150;
    const Maths::Vec3 v(A, A, A);
    const double found {v.length()};
    const double expected {1.732050807568877e150};  // A * sqrt(3)
    const double tolerance {1.0e141};  // relative ~1e-9
    EXPECT_NEAR(found, expected, tolerance);
}
