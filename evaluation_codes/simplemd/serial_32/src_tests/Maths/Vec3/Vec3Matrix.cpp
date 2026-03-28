#include <gtest/gtest.h>
#include "Maths/Vec3.hpp"

// ======================= Vec3: matrix * vector Test Suite =======================

// ---- Identity matrix leaves vector unchanged ----
TEST(Vec3MatMul, IdentityX)
{
    const float m[3][3] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    const Maths::Vec3 v(1.2, -3.4, 5.6);
    const Maths::Vec3 r = m * v;
    const float found {r.x};
    const float expected {1.2};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, IdentityY)
{
    const float m[3][3] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    const Maths::Vec3 v(1.2, -3.4, 5.6);
    const Maths::Vec3 r = m * v;
    const float found {r.y};
    const float expected {-3.4};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, IdentityZ)
{
    const float m[3][3] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    const Maths::Vec3 v(1.2, -3.4, 5.6);
    const Maths::Vec3 r = m * v;
    const float found {r.z};
    const float expected {5.6};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Zero matrix results in zero vector ----
TEST(Vec3MatMul, ZeroMatrixX)
{
    const float m[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const float found {r.x};
    const float expected {0.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, ZeroMatrixY)
{
    const float m[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const float found {r.y};
    const float expected {0.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, ZeroMatrixZ)
{
    const float m[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const float found {r.z};
    const float expected {0.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Diagonal scaling matrix ----
TEST(Vec3MatMul, DiagonalScalingX)
{
    const float m[3][3] = {{ 2.0, 0.0, 0.0},
                            { 0.0,-3.0, 0.0},
                            { 0.0, 0.0, 0.5}};
    const Maths::Vec3 v(1.0, -2.0, 4.0);
    const Maths::Vec3 r = m * v;
    const float found {r.x};
    const float expected {2.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, DiagonalScalingY)
{
    const float m[3][3] = {{ 2.0, 0.0, 0.0},
                            { 0.0,-3.0, 0.0},
                            { 0.0, 0.0, 0.5}};
    const Maths::Vec3 v(1.0, -2.0, 4.0);
    const Maths::Vec3 r = m * v;
    const float found {r.y};
    const float expected {6.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, DiagonalScalingZ)
{
    const float m[3][3] = {{ 2.0, 0.0, 0.0},
                            { 0.0,-3.0, 0.0},
                            { 0.0, 0.0, 0.5}};
    const Maths::Vec3 v(1.0, -2.0, 4.0);
    const Maths::Vec3 r = m * v;
    const float found {r.z};
    const float expected {2.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Rotation about Z by +90 degrees: [[0,-1,0],[1,0,0],[0,0,1]] ----
TEST(Vec3MatMul, RotZ90_X)
{
    const float m[3][3] = {{0.0,-1.0,0.0},
                            {1.0, 0.0,0.0},
                            {0.0, 0.0,1.0}};
    const Maths::Vec3 v(1.0, 2.0, 3.0);
    const Maths::Vec3 r = m * v;
    const float found {r.x};
    const float expected {-2.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, RotZ90_Y)
{
    const float m[3][3] = {{0.0,-1.0,0.0},
                            {1.0, 0.0,0.0},
                            {0.0, 0.0,1.0}};
    const Maths::Vec3 v(1.0, 2.0, 3.0);
    const Maths::Vec3 r = m * v;
    const float found {r.y};
    const float expected {1.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, RotZ90_Z)
{
    const float m[3][3] = {{0.0,-1.0,0.0},
                            {1.0, 0.0,0.0},
                            {0.0, 0.0,1.0}};
    const Maths::Vec3 v(1.0, 2.0, 3.0);
    const Maths::Vec3 r = m * v;
    const float found {r.z};
    const float expected {3.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Permutation matrix swapping Y and Z: [[1,0,0],[0,0,1],[0,1,0]] ----
TEST(Vec3MatMul, SwapYZ_X)
{
    const float m[3][3] = {{1.0,0.0,0.0},
                            {0.0,0.0,1.0},
                            {0.0,1.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const float found {r.x};
    const float expected {7.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, SwapYZ_Y)
{
    const float m[3][3] = {{1.0,0.0,0.0},
                            {0.0,0.0,1.0},
                            {0.0,1.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const float found {r.y};
    const float expected {9.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, SwapYZ_Z)
{
    const float m[3][3] = {{1.0,0.0,0.0},
                            {0.0,0.0,1.0},
                            {0.0,1.0,0.0}};
    const Maths::Vec3 v(7.0, -8.0, 9.0);
    const Maths::Vec3 r = m * v;
    const float found {r.z};
    const float expected {-8.0};
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Arbitrary matrix with decimals ----
TEST(Vec3MatMul, ArbitraryMatrixX)
{
    const float m[3][3] = {{ 1.5,-2.0, 0.5},
                            { 0.0, 3.0,-1.0},
                            { 4.0, 0.25,2.0}};
    const Maths::Vec3 v(-1.0, 2.0, 3.5);
    const Maths::Vec3 r = m * v;
    const float found {r.x};
    const float expected {-3.75};  // -1.5 - 4.0 + 1.75
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, ArbitraryMatrixY)
{
    const float m[3][3] = {{ 1.5,-2.0, 0.5},
                            { 0.0, 3.0,-1.0},
                            { 4.0, 0.25,2.0}};
    const Maths::Vec3 v(-1.0, 2.0, 3.5);
    const Maths::Vec3 r = m * v;
    const float found {r.y};
    const float expected {2.5};    // 0 + 6.0 - 3.5
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, ArbitraryMatrixZ)
{
    const float m[3][3] = {{ 1.5,-2.0, 0.5},
                            { 0.0, 3.0,-1.0},
                            { 4.0, 0.25,2.0}};
    const Maths::Vec3 v(-1.0, 2.0, 3.5);
    const Maths::Vec3 r = m * v;
    const float found {r.z};
    const float expected {3.5};    // -4.0 + 0.5 + 7.0
    const float tolerance {1.0e-12};
    EXPECT_NEAR(found, expected, tolerance);
}

// ---- Large magnitude stress (scaled diagonals) ----
TEST(Vec3MatMul, LargeMagnitudesX)
{
    const float m[3][3] = {{1.0e-3, 0.0,   0.0  },
                            {0.0,    1.0e3, 0.0  },
                            {0.0,    0.0,  -1.0e3}};
    const Maths::Vec3 v(1.0e12, 1.0e-12, 1.0e9);
    const Maths::Vec3 r = m * v;
    const float found {r.x};
    const float expected {1.0e9};
    const float tolerance {1.0e0}; // ~1e-9 relative
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, LargeMagnitudesY)
{
    const float m[3][3] = {{1.0e-3, 0.0,   0.0  },
                            {0.0,    1.0e3, 0.0  },
                            {0.0,    0.0,  -1.0e3}};
    const Maths::Vec3 v(1.0e12, 1.0e-12, 1.0e9);
    const Maths::Vec3 r = m * v;
    const float found {r.y};
    const float expected {1.0e-9};
    const float tolerance {1.0e-18};
    EXPECT_NEAR(found, expected, tolerance);
}

TEST(Vec3MatMul, LargeMagnitudesZ)
{
    const float m[3][3] = {{1.0e-3, 0.0,   0.0  },
                            {0.0,    1.0e3, 0.0  },
                            {0.0,    0.0,  -1.0e3}};
    const Maths::Vec3 v(1.0e12, 1.0e-12, 1.0e9);
    const Maths::Vec3 r = m * v;
    const float found {r.z};
    const float expected {-1.0e12};
    const float tolerance {1.0e3};
    EXPECT_NEAR(found, expected, tolerance);
}
