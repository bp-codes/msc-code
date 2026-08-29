/*********************************************************************************************************************************/
#include <gtest/gtest.h>

#include "Maths/Vec3.hpp"

/*********************************************************************************************************************************/

// ---- Standard basis: i x j = k ----
TEST(Vec3Cross, IxJ_X) {
    const Maths::Vec3 a(1.0, 0.0, 0.0);
    const Maths::Vec3 b(0.0, 1.0, 0.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.x};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, IxJ_Y) {
    const Maths::Vec3 a(1.0, 0.0, 0.0);
    const Maths::Vec3 b(0.0, 1.0, 0.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.y};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, IxJ_Z) {
    const Maths::Vec3 a(1.0, 0.0, 0.0);
    const Maths::Vec3 b(0.0, 1.0, 0.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.z};
    const float expected{1.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Swapped order: j x i = -k ----
TEST(Vec3Cross, JxI_X) {
    const Maths::Vec3 a(0.0, 1.0, 0.0);
    const Maths::Vec3 b(1.0, 0.0, 0.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.x};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, JxI_Y) {
    const Maths::Vec3 a(0.0, 1.0, 0.0);
    const Maths::Vec3 b(1.0, 0.0, 0.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.y};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, JxI_Z) {
    const Maths::Vec3 a(0.0, 1.0, 0.0);
    const Maths::Vec3 b(1.0, 0.0, 0.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.z};
    const float expected{-1.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Parallel vectors give zero ----
TEST(Vec3Cross, ParallelGivesZero_X) {
    const Maths::Vec3 a(2.0, 2.0, 2.0);
    const Maths::Vec3 b(1.0, 1.0, 1.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.x};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, ParallelGivesZero_Y) {
    const Maths::Vec3 a(2.0, 2.0, 2.0);
    const Maths::Vec3 b(1.0, 1.0, 1.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.y};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, ParallelGivesZero_Z) {
    const Maths::Vec3 a(2.0, 2.0, 2.0);
    const Maths::Vec3 b(1.0, 1.0, 1.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.z};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Arbitrary known pair: (1,2,3) x (4,5,6) = (-3, 6, -3) ----
TEST(Vec3Cross, ArbitraryPair_X) {
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(4.0, 5.0, 6.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.x};
    const float expected{-3.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, ArbitraryPair_Y) {
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(4.0, 5.0, 6.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.y};
    const float expected{6.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, ArbitraryPair_Z) {
    const Maths::Vec3 a(1.0, 2.0, 3.0);
    const Maths::Vec3 b(4.0, 5.0, 6.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.z};
    const float expected{-3.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Mixed signs example: (1.5, -2.0, 3.0) x (-4.0, 0.5, 2.0) = (-5.5, -15.0, -7.25) ----
TEST(Vec3Cross, MixedSigns_X) {
    const Maths::Vec3 a(1.5, -2.0, 3.0);
    const Maths::Vec3 b(-4.0, 0.5, 2.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.x};
    const float expected{-5.5};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, MixedSigns_Y) {
    const Maths::Vec3 a(1.5, -2.0, 3.0);
    const Maths::Vec3 b(-4.0, 0.5, 2.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.y};
    const float expected{-15.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, MixedSigns_Z) {
    const Maths::Vec3 a(1.5, -2.0, 3.0);
    const Maths::Vec3 b(-4.0, 0.5, 2.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.z};
    const float expected{-7.25};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Rectangular example: (3,0,0) x (0,4,0) = (0,0,12) ----
TEST(Vec3Cross, AxisScaled_X) {
    const Maths::Vec3 a(3.0, 0.0, 0.0);
    const Maths::Vec3 b(0.0, 4.0, 0.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.x};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, AxisScaled_Y) {
    const Maths::Vec3 a(3.0, 0.0, 0.0);
    const Maths::Vec3 b(0.0, 4.0, 0.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.y};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Cross, AxisScaled_Z) {
    const Maths::Vec3 a(3.0, 0.0, 0.0);
    const Maths::Vec3 b(0.0, 4.0, 0.0);
    const Maths::Vec3 r = a.cross(b);
    const float found{r.z};
    const float expected{12.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}
