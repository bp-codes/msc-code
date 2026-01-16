#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ======================= Vec3: matrix * vector Test Suite =======================

// ---- Identity matrix leaves vector unchanged ----
TEST(Vec3MatMul, IdentityX)
{
    const double m[3][3] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    const Maths::Vec3 v(1.2, -3.4, 5.6);
    const Maths::Vec3 r = m * v;
    const double found {r.x};
    const double expected {1.2};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, IdentityY)
{
    const double m[3][3] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    const Maths::Vec3 v(1.2, -3.4, 5.6);
    const Maths::Vec3 r = m * v;
    const double found {r.y};
    const double expected {-3.4};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, IdentityZ)
{
    const double m[3][3] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    const Maths::Vec3 v(1.2, -3.4, 5.6);
    const Maths::Vec3 r = m * v;
    const double found {r.z};
    const double expected {5.6};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Zero matrix results in zero vector ----
TEST(Vec3MatMul, ZeroMatrixX)
{
    const double m[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const double found {r.x};
    const double expected {0.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, ZeroMatrixY)
{
    const double m[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const double found {r.y};
    const double expected {0.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, ZeroMatrixZ)
{
    const double m[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const double found {r.z};
    const double expected {0.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Diagonal scaling matrix ----
TEST(Vec3MatMul, DiagonalScalingX)
{
    const double m[3][3] = {{ 2.0, 0.0, 0.0},
                            { 0.0,-3.0, 0.0},
                            { 0.0, 0.0, 0.5}};
    const Maths::Vec3 v(1.0, -2.0, 4.0);
    const Maths::Vec3 r = m * v;
    const double found {r.x};
    const double expected {2.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, DiagonalScalingY)
{
    const double m[3][3] = {{ 2.0, 0.0, 0.0},
                            { 0.0,-3.0, 0.0},
                            { 0.0, 0.0, 0.5}};
    const Maths::Vec3 v(1.0, -2.0, 4.0);
    const Maths::Vec3 r = m * v;
    const double found {r.y};
    const double expected {6.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, DiagonalScalingZ)
{
    const double m[3][3] = {{ 2.0, 0.0, 0.0},
                            { 0.0,-3.0, 0.0},
                            { 0.0, 0.0, 0.5}};
    const Maths::Vec3 v(1.0, -2.0, 4.0);
    const Maths::Vec3 r = m * v;
    const double found {r.z};
    const double expected {2.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Rotation about Z by +90 degrees: [[0,-1,0],[1,0,0],[0,0,1]] ----
TEST(Vec3MatMul, RotZ90_X)
{
    const double m[3][3] = {{0.0,-1.0,0.0},
                            {1.0, 0.0,0.0},
                            {0.0, 0.0,1.0}};
    const Maths::Vec3 v(1.0, 2.0, 3.0);
    const Maths::Vec3 r = m * v;
    const double found {r.x};
    const double expected {-2.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, RotZ90_Y)
{
    const double m[3][3] = {{0.0,-1.0,0.0},
                            {1.0, 0.0,0.0},
                            {0.0, 0.0,1.0}};
    const Maths::Vec3 v(1.0, 2.0, 3.0);
    const Maths::Vec3 r = m * v;
    const double found {r.y};
    const double expected {1.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, RotZ90_Z)
{
    const double m[3][3] = {{0.0,-1.0,0.0},
                            {1.0, 0.0,0.0},
                            {0.0, 0.0,1.0}};
    const Maths::Vec3 v(1.0, 2.0, 3.0);
    const Maths::Vec3 r = m * v;
    const double found {r.z};
    const double expected {3.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Permutation matrix swapping Y and Z: [[1,0,0],[0,0,1],[0,1,0]] ----
TEST(Vec3MatMul, SwapYZ_X)
{
    const double m[3][3] = {{1.0,0.0,0.0},
                            {0.0,0.0,1.0},
                            {0.0,1.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const double found {r.x};
    const double expected {7.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, SwapYZ_Y)
{
    const double m[3][3] = {{1.0,0.0,0.0},
                            {0.0,0.0,1.0},
                            {0.0,1.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const double found {r.y};
    const double expected {9.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, SwapYZ_Z)
{
    const double m[3][3] = {{1.0,0.0,0.0},
                            {0.0,0.0,1.0},
                            {0.0,1.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const double found {r.z};
    const double expected {-8.0};
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Arbitrary matrix with decimals ----
TEST(Vec3MatMul, ArbitraryMatrixX)
{
    const double m[3][3] = {{ 1.5,-2.0, 0.5},
                            { 0.0, 3.0,-1.0},
                            { 4.0, 0.25,2.0}};
    const Maths::Vec3 v(-1.0, 2.0, 3.5);
    const Maths::Vec3 r = m * v;
    const double found {r.x};
    const double expected {-3.75};  // -1.5 - 4.0 + 1.75
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, ArbitraryMatrixY)
{
    const double m[3][3] = {{ 1.5,-2.0, 0.5},
                            { 0.0, 3.0,-1.0},
                            { 4.0, 0.25,2.0}};
    const Maths::Vec3 v(-1.0, 2.0, 3.5);
    const Maths::Vec3 r = m * v;
    const double found {r.y};
    const double expected {2.5};    // 0 + 6.0 - 3.5
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, ArbitraryMatrixZ)
{
    const double m[3][3] = {{ 1.5,-2.0, 0.5},
                            { 0.0, 3.0,-1.0},
                            { 4.0, 0.25,2.0}};
    const Maths::Vec3 v(-1.0, 2.0, 3.5);
    const Maths::Vec3 r = m * v;
    const double found {r.z};
    const double expected {3.5};    // -4.0 + 0.5 + 7.0
    const double tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitude stress (scaled diagonals) ----
TEST(Vec3MatMul, LargeMagnitudesX)
{
    const double m[3][3] = {{1.0e-3, 0.0,   0.0  },
                            {0.0,    1.0e3, 0.0  },
                            {0.0,    0.0,  -1.0e3}};
    const Maths::Vec3 v(1.0e12, 1.0e-12, 1.0e9);
    const Maths::Vec3 r = m * v;
    const double found {r.x};
    const double expected {1.0e9};
    const double tolerance {1.0e0}; // ~1e-9 relative
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, LargeMagnitudesY)
{
    const double m[3][3] = {{1.0e-3, 0.0,   0.0  },
                            {0.0,    1.0e3, 0.0  },
                            {0.0,    0.0,  -1.0e3}};
    const Maths::Vec3 v(1.0e12, 1.0e-12, 1.0e9);
    const Maths::Vec3 r = m * v;
    const double found {r.y};
    const double expected {1.0e-9};
    const double tolerance {1.0e-18};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, LargeMagnitudesZ)
{
    const double m[3][3] = {{1.0e-3, 0.0,   0.0  },
                            {0.0,    1.0e3, 0.0  },
                            {0.0,    0.0,  -1.0e3}};
    const Maths::Vec3 v(1.0e12, 1.0e-12, 1.0e9);
    const Maths::Vec3 r = m * v;
    const double found {r.z};
    const double expected {-1.0e12};
    const double tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}
