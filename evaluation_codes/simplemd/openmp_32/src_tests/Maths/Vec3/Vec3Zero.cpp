/*********************************************************************************************************************************/
#include <gtest/gtest.h>

#include "Maths/Vec3.hpp"

/*********************************************************************************************************************************/

TEST(Vec3NZero, Zero1) {
    Maths::Vec3 v{};
    v.zero();
    const float found{v.x};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3NZero, Zero2) {
    Maths::Vec3 v{};
    v.zero();
    const float found{v.y};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3NZero, Zero3) {
    Maths::Vec3 v{};
    v.zero();
    const float found{v.z};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3NZero, Zero4) {
    Maths::Vec3 v{1.0, 2.0, 3.0};
    v.zero();
    const float found{v.x};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3NZero, Zero5) {
    Maths::Vec3 v{1.0, 2.0, 3.0};
    v.zero();
    const float found{v.y};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3NZero, Zero6) {
    Maths::Vec3 v{1.0, 2.0, 3.0};
    v.zero();
    const float found{v.z};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}
