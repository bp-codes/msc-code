// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>

#include "Matrix.hpp"
#include "Error.hpp"
#include "helper.hpp"
#include "json.hpp"

#include <sycl/sycl.hpp>
#include <system_error>

//#include "SyclFunctions.hpp"


constexpr std::size_t TILE = 16;

void dgemm_sycl_tiled(
    sycl::queue& q,
    double alpha,
    const double* A,
    const double* B,
    double beta,
    const double* C,
    std::size_t M,
    std::size_t N,
    std::size_t K,
    double* X)
{
    q.submit([&](sycl::handler& h)
    {
        sycl::local_accessor<double, 2> tileA({TILE, TILE}, h);
        sycl::local_accessor<double, 2> tileB({TILE, TILE}, h);

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>((M + TILE - 1)/TILE * TILE,
                               (N + TILE - 1)/TILE * TILE),
                sycl::range<2>(TILE, TILE)),
            [=](sycl::nd_item<2> item)
            {
                const std::size_t row = item.get_global_id(0);
                const std::size_t col = item.get_global_id(1);

                const std::size_t local_row = item.get_local_id(0);
                const std::size_t local_col = item.get_local_id(1);

                double sum = 0.0;

                for (std::size_t t = 0; t < K; t += TILE)
                {
                    // load tiles
                    if (row < M && (t + local_col) < K)
                        tileA[local_row][local_col] = A[row * K + t + local_col];
                    else
                        tileA[local_row][local_col] = 0.0;

                    if (col < N && (t + local_row) < K)
                        tileB[local_row][local_col] = B[(t + local_row) * N + col];
                    else
                        tileB[local_row][local_col] = 0.0;

                    item.barrier(sycl::access::fence_space::local_space);

                    for (std::size_t k = 0; k < TILE; ++k)
                    {
                        sum += tileA[local_row][k] * tileB[k][local_col];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (row < M && col < N)
                {
                    X[row * N + col] =
                        alpha * sum + beta * C[row * N + col];
                }
            });
    }).wait();
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
    
    double test_time_seconds = std::atof(argv[1]);

    const std::size_t M = std::atoi(argv[2]);  // rows of A and C
    const std::size_t N = std::atoi(argv[3]);  // cols of B and C
    const std::size_t K = std::atoi(argv[4]);  // cols of A / rows of B

    std::string_view device_string = "GPU";
    if (argc >= 6)
    {
        device_string = argv[5];

        if (device_string != "GPU" && device_string != "CPU")
        {
            THROW_INVALID_ARGUMENT("device must be GPU or CPU");
        }
    }

    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);  // [0.0, 1.0)

    // Set up Matrices
    Matrix<double>A {M, K};
    Matrix<double>B {K, N};
    Matrix<double>C {M, N};

    // Fill with random data
    const double k = dist(rng);
    const double l = dist(rng);
    A.random_fill(rng, dist);
    B.random_fill(rng, dist);
    C.random_fill(rng, dist);

    Matrix<double>X_expected = dgemm_serial(k, A, B, l, C);
    const auto expected_value = helper::check_sum(X_expected.vector());
    std::cout << expected_value << std::endl;
    

    // ======= Calculation Starts ========
    
    // Setup
    const auto t0 = std::chrono::steady_clock::now();

    // Set up queue on selected device (CPU or GPU)
    sycl::queue q =
    (device_string == "CPU")
    ? sycl::queue{sycl::cpu_selector_v}
    : sycl::queue{sycl::gpu_selector_v};

    // Report to user that device is being used
    std::cerr << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";


    // Allocate device memory once
    double* dA = sycl::malloc_device<double>(M * K, q);
    double* dB = sycl::malloc_device<double>(K * N, q);
    double* dC = sycl::malloc_device<double>(M * N, q);
    double* dX = sycl::malloc_device<double>(M * N, q);

    // copy once
    q.memcpy(dA, A.data(), sizeof(double) * M * K).wait();
    q.memcpy(dB, B.data(), sizeof(double) * K * N).wait();
    q.memcpy(dC, C.data(), sizeof(double) * M * N).wait();


    // Do calculation
    const auto t1 = std::chrono::steady_clock::now();
    const auto deadline = t1 + std::chrono::duration<double>(test_time_seconds);
    std::uint64_t iters = 0;

    Matrix<double>X {M, N};

    // Test starts
    do 
    {
        dgemm_sycl_tiled(q, k, dA, dB, l, dC, M, N, K, dX);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);

    // Test ends
    q.memcpy(X.data(), dX, sizeof(double) * M * N).wait();

    // Clean up
    const auto t2 = std::chrono::steady_clock::now();

    // Free memory
    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    sycl::free(dX, q);

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
    const auto method {std::string("Parallel Sycl " + matrix_size)};
    const auto comments {std::string("operation:") + std::string(operation_string)};

    // Output
    {

        const std::string base_file_name = "results/parallel_sycl_" + std::string(operation_string);
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = operation_string;
        j["comments"] = comments;
        j["threads"] = 1;
        j["precision"] = "64";
        j["device"] = device_string;
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
