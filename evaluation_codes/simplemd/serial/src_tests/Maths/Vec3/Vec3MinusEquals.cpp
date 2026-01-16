/*********************************************************************************************************************************/
#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"
/*********************************************************************************************************************************/



// ---- Basic subtraction ----
TEST(Vec3MinusEquals, SubtractsComponentsX)
{
    Maths::Vec3 a(5.0, 7.0, 9.0);
    const Maths::Vec3 b(1.0, 2.0, 3.0);
    a -= b;
    const double found {a.x};
    const double expected {4.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MinusEquals, SubtractsComponentsY)
{
    Maths::Vec3 a(5.0, 7.0, 9.0);
    const Maths::Vec3 b(1.0, 2.0, 3.0);
    a -= b;
    const double found {a.y};
    const double expected {5.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MinusEquals, SubtractsComponentsZ)
{
    Maths::Vec3 a(5.0, 7.0, 9.0);
    const Maths::Vec3 b(1.0, 2.0, 3.0);
    a -= b;
    const double found {a.z};
    const double expected {6.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- With negatives ----
TEST(Vec3MinusEquals, WorksWithNegativeValuesX)
{
    Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    a -= b;
    const double found {a.x};
    const double expected {-6.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MinusEquals, WorksWithNegativeValuesY)
{
    Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    a -= b;
    const double found {a.y};
    const double expected {4.0};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MinusEquals, WorksWithNegativeValuesZ)
{
    Maths::Vec3 a(-1.5, 2.0, -3.5);
    const Maths::Vec3 b(4.5, -2.0, 1.0);
    a -= b;
    const double found {a.z};
    const double expected {-4.5};
    const double tolerance {1.0e-9};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitude values ----
TEST(Vec3MinusEquals, WorksWithLargeMagnitudesX)
{
    const double A = 1.0e12;
    Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    a -= b;
    const double found {a.x};
    const double expected {2.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MinusEquals, WorksWithLargeMagnitudesY)
{
    const double A = 1.0e12;
    Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    a -= b;
    const double found {a.y};
    const double expected {0.0};
    const double tolerance {1.0e-3};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MinusEquals, WorksWithLargeMagnitudesZ)
{
    const double A = 1.0e12;
    Maths::Vec3 a(A, A, A);
    const Maths::Vec3 b(-A, A, -A);
    a -= b;
    const double found {a.z};
    const double expected {2.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Returns reference to self ----
TEST(Vec3MinusEquals, ReturnsReferenceToSelf)
{
    Maths::Vec3 a(1.0, 2.0, 3.0);
    Maths::Vec3 b(4.0, 5.0, 6.0);
    Maths::Vec3& ref = (a -= b);
    EXPECT_EQ(&ref, &a);
}
