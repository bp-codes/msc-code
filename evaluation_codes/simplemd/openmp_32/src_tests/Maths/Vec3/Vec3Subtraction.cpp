/*********************************************************************************************************************************/
#include <gtest/gtest.h>

#include "Maths/Vec3.hpp"

/*********************************************************************************************************************************/

// ---- Simple subtraction ----
TEST(Vec3Subtraction, SubtractsPositiveComponentsX) {
    const Maths::Vec3 a(5.0, 7.0, 9.0);
    const Maths::Vec3 b(1.0, 2.0, 3.0);
    const Maths::Vec3 r = a - b;
    const float found{r.x};
    const float expected{4.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Subtraction, SubtractsPositiveComponentsY) {
    const Maths::Vec3 a(5.0, 7.0, 9.0);
    const Maths::Vec3 b(1.0, 2.0, 3.0);
    const Maths::Vec3 r = a - b;
    const float found{r.y};
    const float expected{5.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Subtraction, SubtractsPositiveComponentsZ) {
    const Maths::Vec3 a(5.0, 7.0, 9.0);
    const Maths::Vec3 b(1.0, 2.0, 3.0);
    const Maths::Vec3 r = a - b;
    const float found{r.z};
    const float expected{6.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Mixed positive/negative values ----
TEST(Vec3Subtraction, SubtractsWithNegativesX) {
    const Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    const Maths::Vec3 r = a - b;
    const float found{r.x};
    const float expected{-6.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Subtraction, SubtractsWithNegativesY) {
    const Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    const Maths::Vec3 r = a - b;
    const float found{r.y};
    const float expected{4.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Subtraction, SubtractsWithNegativesZ) {
    const Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    const Maths::Vec3 r = a - b;
    const float found{r.z};
    const float expected{-4.5};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Identical vectors give zero ----
TEST(Vec3Subtraction, IdenticalVectorsX) {
    const Maths::Vec3 a(2.2, -3.3, 4.4);
    const Maths::Vec3 b(2.2, -3.3, 4.4);
    const Maths::Vec3 r = a - b;
    const float found{r.x};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Subtraction, IdenticalVectorsY) {
    const Maths::Vec3 a(2.2, -3.3, 4.4);
    const Maths::Vec3 b(2.2, -3.3, 4.4);
    const Maths::Vec3 r = a - b;
    const float found{r.y};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Subtraction, IdenticalVectorsZ) {
    const Maths::Vec3 a(2.2, -3.3, 4.4);
    const Maths::Vec3 b(2.2, -3.3, 4.4);
    const Maths::Vec3 r = a - b;
    const float found{r.z};
    const float expected{0.0};
    const float tolerance{1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitudes ----
TEST(Vec3Subtraction, WorksWithLargeMagnitudesX) {
    const float A = 1.0e12;
    const Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = a - b;
    const float found{r.x};
    const float expected{2.0e12};
    const float tolerance{1.0e3};  // relative ~1e-9
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Subtraction, WorksWithLargeMagnitudesY) {
    const float A = 1.0e12;
    const Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = a - b;
    const float found{r.y};
    const float expected{0.0};
    const float tolerance{1.0e-3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3Subtraction, WorksWithLargeMagnitudesZ) {
    const float A = 1.0e12;
    const Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    const Maths::Vec3 r = a - b;
    const float found{r.z};
    const float expected{2.0e12};
    const float tolerance{1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}
