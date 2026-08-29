/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



TEST(Vec3IndexOperatorConst, IndexZeroReturnsX)
{
    const Maths::Vec3 v(1.1, 2.2, 3.3);
    const float found {v[0]};
    const float expected {1.1};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3IndexOperatorConst, IndexOneReturnsY)
{
    const Maths::Vec3 v(1.1, 2.2, 3.3);
    const float found {v[1]};
    const float expected {2.2};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3IndexOperatorConst, IndexTwoReturnsZ)
{
    const Maths::Vec3 v(1.1, 2.2, 3.3);
    const float found {v[2]};
    const float expected {3.3};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Reference semantics (addresses must match) ----
TEST(Vec3IndexOperatorConst, AddressOfIndexZeroMatchesX)
{
    const Maths::Vec3 v(4.0, 5.0, 6.0);
    const float* found { &v[0] };
    const float* expected { &v.x };
    EXPECT_EQ(found, expected);
}

TEST(Vec3IndexOperatorConst, AddressOfIndexOneMatchesY)
{
    const Maths::Vec3 v(4.0, 5.0, 6.0);
    const float* found { &v[1] };
    const float* expected { &v.y };
    EXPECT_EQ(found, expected);
}

TEST(Vec3IndexOperatorConst, AddressOfIndexTwoMatchesZ)
{
    const Maths::Vec3 v(4.0, 5.0, 6.0);
    const float* found { &v[2] };
    const float* expected { &v.z };
    EXPECT_EQ(found, expected);
}