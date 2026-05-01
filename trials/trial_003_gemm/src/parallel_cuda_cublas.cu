// openmp.cpp

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "helper/Error.hpp"
#include "helper/Matrix.hpp"
#include "helper/helper.hpp"

#include <nlohmann/json.hpp>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t _e = (call);                                                                   \
        if (_e != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                                               \
    do {                                                                                 \
        cublasStatus_t status = (call);                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                           \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

inline void task(const double* d_A, const double* d_B, const double* d_C, double* d_X,
                 const double k, const double l, const int M, const int N, const int K,
                 cublasHandle_t handle) {
    // X = C
    CUBLAS_CHECK(cublasDcopy(handle, M * N, d_C, 1, d_X, 1));

    // X = k*A*B + l*X
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M,  // rows
                             N,  // cols
                             K, &k, d_A, M, d_B, K, &l, d_X, M));
}

// X = k A * B + l C
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " test_time rows cols\n";
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
    Matrix<double> A{M, K};
    Matrix<double> B{K, N};
    Matrix<double> C{M, N};

    // Check matrix sizes
    if (A.rows() != M || A.cols() != K)
        throw std::invalid_argument("A must be M x K");

    if (B.rows() != K || B.cols() != N)
        throw std::invalid_argument("B must be K x N");

    if (C.rows() != M || C.cols() != N)
        throw std::invalid_argument("C must be M x N");

    // Fill with random data
    const auto k = dist(rng);
    const auto l = dist(rng);
    A.random_fill(rng, dist);
    B.random_fill(rng, dist);
    C.random_fill(rng, dist);

    // ======= Calculation Starts ========

    // Setup
    auto t0 = std::chrono::steady_clock::now();

    // Allocate device memory once
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    double* d_X = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, A.vector().size() * sizeof(double)));  // rows*cols
    CUDA_CHECK(cudaMalloc(&d_B, B.vector().size() * sizeof(double)));  // cols*rows
    CUDA_CHECK(cudaMalloc(&d_C, C.vector().size() * sizeof(double)));  // rows*rows
    CUDA_CHECK(cudaMalloc(&d_X, M * N * sizeof(double)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Copy once (async copies are fine; sync before timing loop if needed)
    auto A_col = A.vector_column_major();
    auto B_col = B.vector_column_major();
    auto C_col = C.vector_column_major();

    CUDA_CHECK(
        cudaMemcpy(d_A, A_col.data(), A_col.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_B, B_col.data(), B_col.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_C, C_col.data(), C_col.size() * sizeof(double), cudaMemcpyHostToDevice));

    // Do calculation
    const auto t1 = std::chrono::steady_clock::now();
    const auto deadline = t1 + std::chrono::duration<double>(test_time_seconds);
    std::uint64_t iters = 0;

    Matrix<double> X{M, N};

    // Test starts
    do {
        // X = C
        task(d_A, d_B, d_C, d_X, k, l, M, N, K, handle);
        CUDA_CHECK(cudaDeviceSynchronize());
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);
    // Test ends
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up
    const auto t2 = std::chrono::steady_clock::now();

    std::vector<double> X_col(M * N);
    CUDA_CHECK(
        cudaMemcpy(X_col.data(), d_X, X_col.size() * sizeof(double), cudaMemcpyDeviceToHost));
    X.load_from_column_major(X_col);

    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_X));

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
    const auto method{std::string("Parallel CUDA cuBLAS " + matrix_size)};
    const auto comments{std::string("operation:") + std::string(operation_string)};

    // Output
    {
        const std::string base_file_name =
            "results/parallel_cuda_cublas_" + matrix_size + "_" + std::string(operation_string);
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = operation_string;
        j["comments"] = comments;
        j["threads"] = 1;
        j["precision"] = "64";
        j["device"] = "GPU";
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
