#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ======================= Vec3::operator/= Test Suite =======================

// ---- Divide by positive scalar ----
TEST(Vec3DivideEquals, DivideByPositiveX)
{
    Maths::Vec3 a(2.0, -4.0, 6.0);
    a /= 2.0;
    const float found {a.x};
    const float expected {1.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideEquals, DivideByPositiveY)
{
    Maths::Vec3 a(2.0, -4.0, 6.0);
    a /= 2.0;
    const float found {a.y};
    const float expected {-2.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideEquals, DivideByPositiveZ)
{
    Maths::Vec3 a(2.0, -4.0, 6.0);
    a /= 2.0;
    const float found {a.z};
    const float expected {3.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Divide by negative scalar ----
TEST(Vec3DivideEquals, DivideByNegativeX)
{
    Maths::Vec3 a(3.0, -6.0, 9.0);
    a /= -3.0;
    const float found {a.x};
    const float expected {-1.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideEquals, DivideByNegativeY)
{
    Maths::Vec3 a(3.0, -6.0, 9.0);
    a /= -3.0;
    const float found {a.y};
    const float expected {2.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideEquals, DivideByNegativeZ)
{
    Maths::Vec3 a(3.0, -6.0, 9.0);
    a /= -3.0;
    const float found {a.z};
    const float expected {-3.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Divide by fractional scalar ----
TEST(Vec3DivideEquals, DivideByFractionX)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a /= 0.5;
    const float found {a.x};
    const float expected {2.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideEquals, DivideByFractionY)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a /= 0.5;
    const float found {a.y};
    const float expected {-4.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideEquals, DivideByFractionZ)
{
    Maths::Vec3 a(1.0, -2.0, 3.0);
    a /= 0.5;
    const float found {a.z};
    const float expected {6.0};
    const float tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitudes ----
TEST(Vec3DivideEquals, DivideLargeValuesX)
{
    const float A = 1.0e12;
    Maths::Vec3 a(A, -A, A);
    a /= 1.0e6;
    const float found {a.x};
    const float expected {1.0e6};
    const float tolerance {1.0e-3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideEquals, DivideLargeValuesY)
{
    const float A = 1.0e12;
    Maths::Vec3 a(A, -A, A);
    a /= 1.0e6;
    const float found {a.y};
    const float expected {-1.0e6};
    const float tolerance {1.0e-3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3DivideEquals, DivideLargeValuesZ)
{
    const float A = 1.0e12;
    Maths::Vec3 a(A, -A, A);
    a /= 1.0e6;
    const float found {a.z};
    const float expected {1.0e6};
    const float tolerance {1.0e-3};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Divide by zero should throw ----
TEST(Vec3DivideEquals, DivideByZeroThrows)
{
    Maths::Vec3 a(1.0, 2.0, 3.0);
    EXPECT_THROW({
        a /= 0.0;
    }, std::invalid_argument);
}

// ---- Returns reference to self ----
TEST(Vec3DivideEquals, ReturnsReferenceToSelf)
{
    Maths::Vec3 a(1.0, 2.0, 3.0);
    Maths::Vec3& ref = (a /= 2.0);
    EXPECT_EQ(&ref, &a);
}
