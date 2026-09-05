/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



TEST(Vec3Constructor, DefaultSetsAllZeroX) 
{
    const Maths::Vec3 v;
    const float found {v.x};
    const float expected {0.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Constructor, DefaultSetsAllZeroY) 
{
    const Maths::Vec3 v;
    const float found {v.y};
    const float expected {0.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Constructor, DefaultSetsAllZeroZ) 
{
    const Maths::Vec3 v;
    const float found {v.z};
    const float expected {0.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}




TEST(Vec3Constructor, BraceInitDefaultX) 
{
    const Maths::Vec3 v {};
    const float found {v.x};
    const float expected {0.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Constructor, BraceInitDefaultY) 
{
    const Maths::Vec3 v {};
    const float found {v.y};
    const float expected {0.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Constructor, BraceInitDefaultZ) 
{
    const Maths::Vec3 v {};
    const float found {v.z};
    const float expected {0.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}





TEST(Vec3Constructor, ValueConstructorSetsX) 
{
    const Maths::Vec3 v(1.5, -2.0, 3.25);
    const float found {v.x};
    const float expected {1.5};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Constructor, ValueConstructorSetsY) 
{
    const Maths::Vec3 v(1.5, -2.0, 3.25);
    const float found {v.y};
    const float expected {-2.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Constructor, ValueConstructorSetsZ) 
{
    const Maths::Vec3 v(1.5, -2.0, 3.25);
    const float found {v.z};
    const float expected {3.25};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}
