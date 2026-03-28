/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



TEST(Vec3Normalize, ZeroVectorStaysZeroX)
{
    const Maths::Vec3 v(0.0, 0.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const float found {n.x};
    const float expected {0.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, ZeroVectorStaysZeroY)
{
    const Maths::Vec3 v(0.0, 0.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const float found {n.y};
    const float expected {0.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, ZeroVectorStaysZeroZ)
{
    const Maths::Vec3 v(0.0, 0.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const float found {n.z};
    const float expected {0.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}


TEST(Vec3Normalize, UnitXStaysSameX)
{
    const Maths::Vec3 v(1.0, 0.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const float found {n.x};
    const float expected {1.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, UnitYStaysSameY)
{
    const Maths::Vec3 v(0.0, 1.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const float found {n.y};
    const float expected {1.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, UnitZStaysSameZ)
{
    const Maths::Vec3 v(0.0, 0.0, -1.0);
    const Maths::Vec3 n = v.normalize();
    const float found {n.z};
    const float expected {-1.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}


TEST(Vec3Normalize, NormalizesThreeFourZeroX)
{
    const Maths::Vec3 v(3.0, 4.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const float found {n.x};
    const float expected {0.6}; // 3/5
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, NormalizesThreeFourZeroY)
{
    const Maths::Vec3 v(3.0, 4.0, 0.0);
    const Maths::Vec3 n = v.normalize();
    const float found {n.y};
    const float expected {0.8}; // 4/5
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Normalize, NormalizedVectorHasLengthOne)
{
    const Maths::Vec3 v(5.0, -2.0, 2.0);
    const Maths::Vec3 n = v.normalize();
    const float found {n.length()};
    const float expected {1.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}


TEST(Vec3Normalize, LargeVectorStillNormalizes)
{
    const float A = 1.0e150;
    const Maths::Vec3 v(A, A, A);
    const Maths::Vec3 n = v.normalize();
    const float found {n.length()};
    const float expected {1.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}
