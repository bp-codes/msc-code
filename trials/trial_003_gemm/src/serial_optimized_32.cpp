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

#include "helper/Matrix.hpp"
#include "helper/Error.hpp"
#include "helper/helper.hpp"
#include <nlohmann/json.hpp>

#include <cblas.h>
#include <omp.h>
#include <immintrin.h>




Matrix<float> dgemm_cblas(
                    const float alpha,
                    const Matrix<float>& A,
                    const Matrix<float>& B,
                    const float beta,
                    const Matrix<float>& C)
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

    Matrix<float> X{M, N};
    X = C;  // so beta * C has the right starting values

    cblas_sgemm(
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



Matrix<float> dgemm_serial(float alpha,
                            const Matrix<float>& A,
                            const Matrix<float>& B,
                            float beta,
                            const Matrix<float>& C)
{
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    // ---- dimension checks ----
    if (B.rows() != K)
        throw std::invalid_argument("B.rows() must equal A.cols()");
    if (C.rows() != M || C.cols() != N)
        throw std::invalid_argument("C must be M x N");

    Matrix<float> X(M, N);

    const auto& A_data = A.vector();
    const auto& B_data = B.vector();
    const auto& C_data = C.vector();
    auto& X_data = X.vector();

    // X = beta * C
    for (std::size_t i = 0; i < M; ++i)
    {
        const auto iN = i * N;
        for (std::size_t j = 0; j < N; ++j)
        {
            const auto idx = iN + j;
            X_data[idx] = beta * C_data[idx];
        }
    }

    constexpr std::size_t BLOCK = 128;

    for (std::size_t ib = 0; ib < M; ib += BLOCK)
    {
        for (std::size_t kb = 0; kb < K; kb += BLOCK)
        {
            for (std::size_t jb = 0; jb < N; jb += BLOCK)
            {
                const std::size_t i_max = std::min(ib + BLOCK, M);
                const std::size_t k_max = std::min(kb + BLOCK, K);
                const std::size_t j_max = std::min(jb + BLOCK, N);

                for (std::size_t i = ib; i < i_max; ++i)
                {
                    const auto iN = i * N;

                    for (std::size_t k = kb; k < k_max; ++k)
                    {
                        const auto kN = k * N;
                        const auto a_ik = alpha * A_data[i * K + k];

                        const auto* Bk = &B_data[kN];
                        auto* Xi = &X_data[iN];

                        #pragma omp simd
                        for (std::size_t j = jb; j < j_max; ++j)
                        {
                            Xi[j] = std::fma(a_ik, Bk[j], Xi[j]);
                        }
                    }
                }
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
    Matrix<float>A {M, K};
    Matrix<float>B {K, N};
    Matrix<float>C {M, N};

    // Fill with random data
    const auto k = static_cast<float>(dist(rng));
    const auto l = static_cast<float>(dist(rng));
    A.random_fill(rng, dist);
    B.random_fill(rng, dist);
    C.random_fill(rng, dist);

    Matrix<float>X_expected = dgemm_cblas(k, A, B, l, C);
    const auto expected_value = helper::check_sum(X_expected.vector());
    std::cout << expected_value << std::endl;



    // ======= Calculation Starts ========

    // Setup
    const auto t0 = std::chrono::steady_clock::now();



    // Do calculation
    const auto t1 = std::chrono::steady_clock::now();
    const auto deadline = t1 + std::chrono::duration<double>(test_time_seconds);
    std::uint64_t iters = 0;

    Matrix<float>X {M, N};

    // Test starts
    do
    {
        X = dgemm_serial(k, A, B, l, C);
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
    const auto method {std::string("Serial Optimized 32 " + matrix_size)};
    const auto comments {std::string("operation:") + std::string(operation_string)};

    // Output
    {

        const std::string base_file_name = "results/serial_optimized_32_" + matrix_size + "_" + std::string(operation_string);
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
