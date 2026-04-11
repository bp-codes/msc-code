// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "Matrix.hpp"
#include "Error.hpp"
#include "helper.hpp"
#include "json.hpp"

#include <cblas.h>





Matrix<double> dgemm_cblas( 
                    const double alpha,
                    const Matrix<double>& A,
                    const Matrix<double>& B,
                    const double beta,
                    const Matrix<double>& C)
{

    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    if (B.rows() != K)
    {
        throw std::invalid_argument("dgemm_cblas: A.cols() must equal B.rows().");
    }

    if (C.rows() != M || C.cols() != N)
    {
        throw std::invalid_argument("dgemm_cblas: C must have shape M x N.");
    }

    Matrix<double> X{M, N};
    X = C;  // so beta * C has the right starting values

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        alpha,
        A.data(), static_cast<int>(K),
        B.data(), static_cast<int>(N),
        beta,
        X.data(), static_cast<int>(N));

    return X;
}



Matrix<double> dgemm_serial(double alpha,
                            const Matrix<double>& A,
                            const Matrix<double>& B,
                            double beta,
                            const Matrix<double>& C)
{
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    // ---- dimension checks ----
    if (B.rows() != K)
        throw std::invalid_argument("B.rows() must equal A.cols()");
    if (C.rows() != M || C.cols() != N)
        throw std::invalid_argument("C must be M x N");

    Matrix<double> X(M, N);

    // ---- X = beta * C ----
    for (std::size_t i = 0; i < M; ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            X(i, j) = beta * C(i, j);
        }
    }

    // ---- X += alpha * A * B ----
    for (std::size_t i = 0; i < M; ++i)
    {
        for (std::size_t k = 0; k < K; ++k)
        {
            const double a_ik = alpha * A(i, k);

            for (std::size_t j = 0; j < N; ++j)
            {
                X(i, j) += a_ik * B(k, j);
            }
        }
    }

    return X;
}






// X = k A * B + l C
int main(int argc, char** argv) 
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " test_time rows cols\n";
        return 1;
    }
    
    double test_time = std::atof(argv[1]);

    const std::size_t M = std::atoi(argv[2]);  // rows of A and C
    const std::size_t N = std::atoi(argv[3]);  // cols of B and C
    const std::size_t K = std::atoi(argv[4]);  // cols of A / rows of B

    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);  // [0.0, 1.0)

    // Set up Matrices
    Matrix<double>A {M, K};
    Matrix<double>B {K, N};
    Matrix<double>C {M, N};

    // Fill with random data
    double k = dist(rng);
    double l = dist(rng);
    A.random_fill(rng, dist);
    B.random_fill(rng, dist);
    C.random_fill(rng, dist);

    Matrix X_expected = dgemm_cblas(k, A, B, l, C);
    double expected_value = helper::check_sum(X_expected.vector());
    std::cout << expected_value << std::endl;

    

    // ======= Calculation Starts ========
    
    // Setup
    auto t0 = std::chrono::steady_clock::now();



    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    Matrix<double>X {M, N};

    // Test starts
    do 
    {
        X = dgemm_serial(k, A, B, l, C);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);
    // Test ends


    // Clean up
    auto t2 = std::chrono::steady_clock::now();


    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    double calculated_value = helper::check_sum(X.vector());


    double time_setup = std::chrono::duration<double>(t1 - t0).count();
    double time_calc = std::chrono::duration<double>(t2 - t1).count();
    double time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    double time_total = std::chrono::duration<double>(t3 - t0).count();
    double time_per_iteration = time_calc / iters;


    std::string method {"Serial"};
    std::string comments {"operation:DGEMM"};
    bool passed_check = std::abs(calculated_value - expected_value) < 1.0e-9;

    std::cout << method << "," 
              << expected_value << "," 
              << calculated_value << "," 
              << iters << "," 
              << time_per_iteration << "," 
              << time_setup << "," 
              << time_calc << "," 
              << time_cleanup << "," 
              << time_total << "," 
              << passed_check << "," 
              << comments << "" 
              << std::endl;
    

    return 0;
}
