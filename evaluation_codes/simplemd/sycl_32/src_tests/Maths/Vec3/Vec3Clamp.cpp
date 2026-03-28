/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



// ---- Default clamp [0,1]: values within range stay the same ----
TEST(Vec3Clamp, ValuesWithinRangeStaySameX)
{
    const Maths::Vec3 v(0.5, 0.25, 0.75);
    const Maths::Vec3 r = v.clamp();
    const double found {r.x};
    const double expected {0.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Clamp, ValuesWithinRangeStaySameY)
{
    const Maths::Vec3 v(0.5, 0.25, 0.75);
    const Maths::Vec3 r = v.clamp();
    const double found {r.y};
    const double expected {0.25};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Clamp, ValuesWithinRangeStaySameZ)
{
    const Maths::Vec3 v(0.5, 0.25, 0.75);
    const Maths::Vec3 r = v.clamp();
    const double found {r.z};
    const double expected {0.75};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Default clamp [0,1]: below/above limits are clamped ----
TEST(Vec3Clamp, BelowMinimumClampedX)
{
    const Maths::Vec3 v(-0.5, 0.0, 0.0);
    const Maths::Vec3 r = v.clamp();
    const double found {r.x};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Clamp, AboveMaximumClampedY)
{
    const Maths::Vec3 v(0.0, 1.5, 0.0);
    const Maths::Vec3 r = v.clamp();
    const double found {r.y};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Custom symmetric range [-1, 1] ----
TEST(Vec3Clamp, CustomRangeSymmetricClampsZ)
{
    const Maths::Vec3 v(0.0, 0.0, 5.0);
    const Maths::Vec3 r = v.clamp(-1.0, 1.0);
    const double found {r.z};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Clamp, CustomRangeSymmetricClampsNegativeX)
{
    const Maths::Vec3 v(-5.0, 0.0, 0.0);
    const Maths::Vec3 r = v.clamp(-1.0, 1.0);
    const double found {r.x};
    const double expected {-1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Clamp, CustomRangeSymmetricKeepsInsideY)
{
    const Maths::Vec3 v(0.0, 0.5, 0.0);
    const Maths::Vec3 r = v.clamp(-1.0, 1.0);
    const double found {r.y};
    const double expected {0.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Custom asymmetric range [2, 5] ----
TEST(Vec3Clamp, AsymmetricRangeClampsBelowMinX)
{
    const Maths::Vec3 v(1.0, 3.0, 4.0);
    const Maths::Vec3 r = v.clamp(2.0, 5.0);
    const double found {r.x};
    const double expected {2.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Clamp, AsymmetricRangeClampsAboveMaxY)
{
    const Maths::Vec3 v(0.0, 6.0, 0.0);
    const Maths::Vec3 r = v.clamp(2.0, 5.0);
    const double found {r.y};
    const double expected {5.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Clamp, AsymmetricRangeKeepsInsideZ)
{
    const Maths::Vec3 v(0.0, 0.0, 3.5);
    const Maths::Vec3 r = v.clamp(2.0, 5.0);
    const double found {r.z};
    const double expected {3.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}
