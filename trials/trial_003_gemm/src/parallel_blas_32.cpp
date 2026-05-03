/**
 * @file serial.cpp
 * @brief
 *
 * @author Ben Palmer
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026 Ben Palmer
 * SPDX-License-Identifier: MIT
 */

#include <cblas.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <string>

#include "helper/Error.hpp"
#include "helper/Matrix.hpp"
#include "helper/helper.hpp"

#include <nlohmann/json.hpp>

Matrix<float> task(const float alpha, const Matrix<float>& A, const Matrix<float>& B,
                   const float beta, const Matrix<float>& C) {
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    if (B.rows() != K) {
        throw std::invalid_argument("dgemm_cblas: A.cols() must equal B.rows().");
    }

    if (C.rows() != M || C.cols() != N) {
        throw std::invalid_argument("dgemm_cblas: C must have shape M x N.");
    }

    Matrix<float> X{M, N};
    X = C;  // so beta * C has the right starting values

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M), static_cast<int>(N),
                static_cast<int>(K), alpha, A.data(), static_cast<int>(K), B.data(),
                static_cast<int>(N), beta, X.data(), static_cast<int>(N));

    return X;
}

// X = k A * B + l C
int main(int argc, char** argv) {
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
    Matrix<float> A{M, K};
    Matrix<float> B{K, N};
    Matrix<float> C{M, N};

    // Fill with random data
    const auto k = static_cast<float>(dist(rng));
    const auto l = dist(rng);
    A.random_fill(rng, dist);
    B.random_fill(rng, dist);
    C.random_fill(rng, dist);

    // ======= Calculation Starts ========

    // Setup
    const auto t0 = std::chrono::steady_clock::now();

    // Do calculation
    const auto t1 = std::chrono::steady_clock::now();
    const auto deadline = t1 + std::chrono::duration<double>(test_time_seconds);
    std::uint64_t iters = 0;

    Matrix<float> X{M, N};

    // Test starts
    do {
        X = task(k, A, B, l, C);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);
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

    const std::string operation_string = "gemm";

    const auto matrix_size{std::to_string(M) + "x" + std::to_string(K) + "_by_" +
                           std::to_string(K) + "x" + std::to_string(N)};
    const auto method{std::string("Parallel BLAS 32")};
    const auto comments{std::string("operation:") + std::string(operation_string)};

    // Output
    {
        const std::string base_file_name =
            "results/parallel_blas_32_" + matrix_size + "_" + std::string(operation_string);
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = matrix_size;
        j["comments"] = comments;
        j["threads"] = 1;
        j["precision"] = "32";
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
        j["calculated_value"] = helper::to_string_precise(calculated_value);
        j["values"] = helper::to_string_precise_vector(X.vector());

        // Memory
        j["max_rss_kb"] = helper::max_rss_kb();

        std::ofstream out(json_file);
        if (!out) {
            throw std::runtime_error("Failed to open output JSON file.");
        }

        // Save JSON file.
        out << std::setw(2) << j << '\n';
    }

    return 0;
}
