/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/




TEST(Vec3IndexOperator, IndexZeroReturnsX) 
{
    Maths::Vec3 v(1.1, 2.2, 3.3);
    const double found {v[0]};
    const double expected {1.1};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3IndexOperator, IndexOneReturnsY) 
{
    Maths::Vec3 v(1.1, 2.2, 3.3);
    const double found {v[1]};
    const double expected {2.2};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3IndexOperator, IndexTwoReturnsZ) 
{
    Maths::Vec3 v(1.1, 2.2, 3.3);
    const double found {v[2]};
    const double expected {3.3};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3IndexOperator, ModifyXThroughIndex) 
{
    Maths::Vec3 v(1.1, 2.2, 3.3);
    v[0] = 9.9;
    const double found {v.x};
    const double expected {9.9};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3IndexOperator, ModifyYThroughIndex) 
{
    Maths::Vec3 v(1.1, 2.2, 3.3);
    v[1] = -7.7;
    const double found {v.y};
    const double expected {-7.7};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3IndexOperator, ModifyZThroughIndex) 
{
    Maths::Vec3 v(1.1, 2.2, 3.3);
    v[2] = 42.0;
    const double found {v.z};
    const double expected {42.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}
