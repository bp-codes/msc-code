/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



TEST(Vec3IndexOperatorConst, IndexZeroReturnsX)
{
    const Maths::Vec3 v(1.1, 2.2, 3.3);
    const double found {v[0]};
    const double expected {1.1};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3IndexOperatorConst, IndexOneReturnsY)
{
    const Maths::Vec3 v(1.1, 2.2, 3.3);
    const double found {v[1]};
    const double expected {2.2};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3IndexOperatorConst, IndexTwoReturnsZ)
{
    const Maths::Vec3 v(1.1, 2.2, 3.3);
    const double found {v[2]};
    const double expected {3.3};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Reference semantics (addresses must match) ----
TEST(Vec3IndexOperatorConst, AddressOfIndexZeroMatchesX)
{
    const Maths::Vec3 v(4.0, 5.0, 6.0);
    const double* found { &v[0] };
    const double* expected { &v.x };
    EXPECT_EQ(found, expected);
}

TEST(Vec3IndexOperatorConst, AddressOfIndexOneMatchesY)
{
    const Maths::Vec3 v(4.0, 5.0, 6.0);
    const double* found { &v[1] };
    const double* expected { &v.y };
    EXPECT_EQ(found, expected);
}

TEST(Vec3IndexOperatorConst, AddressOfIndexTwoMatchesZ)
{
    const Maths::Vec3 v(4.0, 5.0, 6.0);
    const double* found { &v[2] };
    const double* expected { &v.z };
    EXPECT_EQ(found, expected);
}