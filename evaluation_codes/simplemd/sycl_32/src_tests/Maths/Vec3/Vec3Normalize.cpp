/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



TEST(Vec3Normalize, ZeroVectorStaysZeroX)
{
    const Maths::Vec3 v(0.0, 0.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const double found {n.x};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, ZeroVectorStaysZeroY)
{
    const Maths::Vec3 v(0.0, 0.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const double found {n.y};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, ZeroVectorStaysZeroZ)
{
    const Maths::Vec3 v(0.0, 0.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const double found {n.z};
    const double expected {0.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}


TEST(Vec3Normalize, UnitXStaysSameX)
{
    const Maths::Vec3 v(1.0, 0.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const double found {n.x};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, UnitYStaysSameY)
{
    const Maths::Vec3 v(0.0, 1.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const double found {n.y};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, UnitZStaysSameZ)
{
    const Maths::Vec3 v(0.0, 0.0, -1.0);
    const Maths::Vec3 n = v.normalize();
    const double found {n.z};
    const double expected {-1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}


TEST(Vec3Normalize, NormalizesThreeFourZeroX)
{
    const Maths::Vec3 v(3.0, 4.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const double found {n.x};
    const double expected {0.6}; // 3/5
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, NormalizesThreeFourZeroY)
{
    const Maths::Vec3 v(3.0, 4.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const double found {n.y};
    const double expected {0.8}; // 4/5
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, NormalizedVectorHasLengthOne)
{
    const Maths::Vec3 v(5.0, -2.0, 2.0);
    const Maths::Vec3 n = v.normalize();
    const double found {n.length()};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}


TEST(Vec3Normalize, LargeVectorStillNormalizes)
{
    const double A = 1.0e150;
    const Maths::Vec3 v(A, A, A);
    const Maths::Vec3 n = v.normalize();
    const double found {n.length()};
    const double expected {1.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}
