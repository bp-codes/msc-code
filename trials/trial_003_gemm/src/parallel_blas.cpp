// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
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
        std::cerr << "Usage: " << argv[0] << " test_time_seconds rows cols\n";
        return 1;
    }
    
    const double test_time_seconds = std::atof(argv[1]);

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
    const auto k = dist(rng);
    const auto l = dist(rng);
    A.random_fill(rng, dist);
    B.random_fill(rng, dist);
    C.random_fill(rng, dist);

    Matrix<double>X_expected = dgemm_cblas(k, A, B, l, C);
    const auto expected_value = helper::check_sum(X_expected.vector());
    std::cout << expected_value << std::endl;

    

    // ======= Calculation Starts ========
    
    // Setup
    const auto t0 = std::chrono::steady_clock::now();



    // Do calculation
    const auto t1 = std::chrono::steady_clock::now();
    const auto deadline = t1 + std::chrono::duration<double>(test_time_seconds);
    std::uint64_t iters = 0;

    Matrix<double>X {M, N};

    // Test starts
    do 
    {
        X = dgemm_cblas(k, A, B, l, C);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);
    // Test ends


    // Clean up
    const auto t2 = std::chrono::steady_clock::now();


    // Actual end time
    const auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    const auto calculated_value = helper::check_sum(X.vector());


    const auto time_setup = std::chrono::duration<double>(t1 - t0).count();
    const auto time_calc = std::chrono::duration<double>(t2 - t1).count();
    const auto time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    const auto time_total = std::chrono::duration<double>(t3 - t0).count();
    const auto time_per_iteration = time_calc / iters;

    const auto passed_check {std::abs(calculated_value - expected_value) < 1.0e-9};
    const std::string operation_string = "gemm"; 

    const auto matrix_size {std::to_string(M) + "x" + std::to_string(K) + "_by_" + std::to_string(K) + "x" + std::to_string(N)};
    const auto method {std::string("Parallel BLAS " + matrix_size)};
    const auto comments {std::string("operation:") + std::string(operation_string)};

    // Output
    {

        const std::string base_file_name = "results/parallel_blas_" + std::string(operation_string);
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = operation_string;
        j["comments"] = comments;
        j["threads"] = 1;
        j["precision"] = "64";
        j["device"] = "CPU";
        j["M"] = M;
        j["N"] = N;
        j["K"] = K;

        // Iteration/timing            
        j["test_time_seconds"] = test_time_seconds;
        j["iterations"] = iters;
        j["time_per_iteration"] = time_per_iteration;
        j["time_setup"] = time_setup;
        j["time_calc"] = time_calc;
        j["time_cleanup"] = time_cleanup;
        j["time_total"] = time_total;
        
        // Values
        j["expected_value"] = helper::to_string_precise(expected_value);
        j["calculated_value"] = helper::to_string_precise(calculated_value);;
        j["difference"] = helper::to_string_precise(expected_value - calculated_value);
        j["passed_check"] = passed_check;
        j["values"] = helper::to_string_precise_vector(X.vector());

        // Memory
        j["max_rss_kb"] = helper::max_rss_kb();

        std::ofstream out(json_file);
        if (!out)
        {
            throw std::runtime_error("Failed to open output JSON file.");
        }

        // Save JSON file.
        out << std::setw(2) << j << '\n';
    }

    return 0;
}
