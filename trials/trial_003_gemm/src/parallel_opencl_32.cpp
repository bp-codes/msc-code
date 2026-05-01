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

#define CL_TARGET_OPENCL_VERSION 120
#ifndef CL_PLATFORM_NOT_FOUND_KHR
#define CL_PLATFORM_NOT_FOUND_KHR -1001
#endif

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <system_error>
#include <vector>
#include <string>

#include "helper/Error.hpp"
#include "helper/Matrix.hpp"
#include "helper/helper.hpp"

#include <nlohmann/json.hpp>

// OpenCL kernel code
// ======================================================

const char* opencl_kernel_code = R"CLC(
#define TILE 16

__kernel void gemm(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global const float* C,
    __global float* X)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float sum = 0.0f;

    // Loop over tiles in K
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        int kA = t * TILE + local_col;
        int kB = t * TILE + local_row;

        // Load tile of A
        if (row < M && kA < K)
            As[local_row][local_col] = A[row * K + kA];
        else
            As[local_row][local_col] = 0.0f;

        // Load tile of B
        if (kB < K && col < N)
            Bs[local_row][local_col] = B[kB * N + col];
        else
            Bs[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial product
        for (int k = 0; k < TILE; ++k)
        {
            sum += As[local_row][k] * Bs[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N)
    {
        X[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
)CLC";

// OpenCL check
// ======================================================

void opencl_check(cl_int status, const char* message) {
    if (status != CL_SUCCESS) {
        std::ostringstream oss;
        oss << message << " (OpenCL error " << status << ")";
        THROW_RUNTIME_ERROR(oss.str());
    }
}

// OpenCL context
//======================================================

struct OpenCLContext {
    cl_platform_id platform{};
    cl_device_id device{};
    cl_context context{};
    cl_command_queue queue{};
    cl_program program{};
    cl_kernel kernel{};

    cl_mem input_buf{};
    cl_mem matrix_a{};
    cl_mem matrix_b{};
    cl_mem matrix_c{};
    cl_mem matrix_x{};
};

// Setup
//======================================================

void opencl_setup(OpenCLContext& ctx, const Matrix<float>& A, const Matrix<float>& B,
                  const Matrix<float>& C, const Matrix<float>& X, std::string_view device_string) {
    cl_int err;

    // ---------------------------
    // 1. Get platforms
    // ---------------------------
    cl_uint platform_count{0};
    cl_int status = clGetPlatformIDs(0, nullptr, &platform_count);

    if (status == CL_PLATFORM_NOT_FOUND_KHR || platform_count == 0) {
        THROW_RUNTIME_ERROR("No OpenCL platforms found.");
    }

    std::vector<cl_platform_id> platforms(platform_count);
    opencl_check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
                 "clGetPlatformIDs failed");

    // ---------------------------
    // 2. Select device type
    // ---------------------------
    const cl_device_type requested_type =
        (device_string == "CPU") ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;

    // ---------------------------
    // 3. Find device
    // ---------------------------
    ctx.device = nullptr;
    ctx.platform = nullptr;

    for (const auto& platform : platforms) {
        cl_uint device_count{0};
        cl_int device_status = clGetDeviceIDs(platform, requested_type, 0, nullptr, &device_count);

        if (device_status == CL_SUCCESS && device_count > 0) {
            std::vector<cl_device_id> devices(device_count);
            opencl_check(
                clGetDeviceIDs(platform, requested_type, device_count, devices.data(), nullptr),
                "clGetDeviceIDs failed");

            ctx.device = devices.front();
            ctx.platform = platform;
            break;
        }
    }

    if (!ctx.device) {
        std::ostringstream oss;
        oss << "No suitable OpenCL " << device_string << " device found.";
        THROW_RUNTIME_ERROR(oss.str());
    }

    // ---------------------------
    // 4. Create context + queue
    // ---------------------------
    ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &err);
    opencl_check(err, "clCreateContext failed");

    ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, &err);
    opencl_check(err, "clCreateCommandQueue failed");

    // ---------------------------
    // 5. Build program
    // ---------------------------
    const char* src = opencl_kernel_code;

    ctx.program = clCreateProgramWithSource(ctx.context, 1, &src, nullptr, &err);
    opencl_check(err, "clCreateProgramWithSource failed");

    err = clBuildProgram(ctx.program, 1, &ctx.device, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        std::vector<char> log(log_size);
        clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                              nullptr);

        std::cerr << "Build log:\n" << log.data() << "\n";
        opencl_check(err, "clBuildProgram failed");
    }

    ctx.kernel = clCreateKernel(ctx.program, "gemm", &err);
    opencl_check(err, "clCreateKernel failed");

    // ---------------------------
    // 6. Buffers
    // ---------------------------
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    auto* a_ptr = const_cast<float*>(A.data());
    ctx.matrix_a = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * M * K, reinterpret_cast<void*>(a_ptr), &err);
    opencl_check(err, "buffer A failed");

    auto* b_ptr = const_cast<float*>(B.data());
    ctx.matrix_b = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * K * N, reinterpret_cast<void*>(b_ptr), &err);
    opencl_check(err, "buffer B failed");

    auto* c_ptr = const_cast<float*>(C.data());
    ctx.matrix_c = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * M * N, reinterpret_cast<void*>(c_ptr), &err);
    opencl_check(err, "buffer C failed");

    ctx.matrix_x =
        clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, nullptr, &err);
    opencl_check(err, "buffer X failed");
}

void task(OpenCLContext& ctx, float alpha, const Matrix<float>& A, const Matrix<float>& B,
          float beta, const Matrix<float>& C, Matrix<float>& X) {
    cl_int err;

    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    // Set kernel args
    opencl_check(clSetKernelArg(ctx.kernel, 0, sizeof(int), &M), "arg0");
    opencl_check(clSetKernelArg(ctx.kernel, 1, sizeof(int), &N), "arg1");
    opencl_check(clSetKernelArg(ctx.kernel, 2, sizeof(int), &K), "arg2");
    opencl_check(clSetKernelArg(ctx.kernel, 3, sizeof(float), &alpha), "arg3");
    opencl_check(clSetKernelArg(ctx.kernel, 4, sizeof(cl_mem), &ctx.matrix_a), "arg4");
    opencl_check(clSetKernelArg(ctx.kernel, 5, sizeof(cl_mem), &ctx.matrix_b), "arg5");
    opencl_check(clSetKernelArg(ctx.kernel, 6, sizeof(float), &beta), "arg6");
    opencl_check(clSetKernelArg(ctx.kernel, 7, sizeof(cl_mem), &ctx.matrix_c), "arg7");
    opencl_check(clSetKernelArg(ctx.kernel, 8, sizeof(cl_mem), &ctx.matrix_x), "arg8");

    constexpr size_t TILE = 16;

    size_t local[2] = {TILE, TILE};
    size_t global[2] = {(static_cast<std::size_t>(M) + TILE - 1)
                        / TILE * TILE, (static_cast<std::size_t>(N) + TILE - 1) / TILE * TILE};

    opencl_check(clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 2, nullptr, global, local, 0,
                                        nullptr, nullptr),
                 "enqueue kernel failed");

    clFinish(ctx.queue);
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

    std::string_view device_string = "GPU";
    if (argc >= 6) {
        device_string = argv[5];

        if (device_string != "GPU" && device_string != "CPU") {
            THROW_INVALID_ARGUMENT("device must be GPU or CPU");
        }
    }

    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);  // [0.0, 1.0)

    // Set up Matrices
    Matrix<float> A{M, K};
    Matrix<float> B{K, N};
    Matrix<float> C{M, N};

    // Fill with random data
    const auto k = static_cast<float>(dist(rng));
    const auto l = static_cast<float>(dist(rng));
    A.random_fill(rng, dist);
    B.random_fill(rng, dist);
    C.random_fill(rng, dist);

    // ======= Calculation Starts ========

    // Setup
    const auto t0 = std::chrono::steady_clock::now();

    Matrix<float> X{M, N};

    OpenCLContext ctx;
    opencl_setup(ctx, A, B, C, X, device_string);

    // Do calculation
    const auto t1 = std::chrono::steady_clock::now();
    const auto deadline = t1 + std::chrono::duration<double>(test_time_seconds);
    std::uint64_t iters = 0;

    // Test starts
    do {
        // X = gemm_parallel(k, A, B, l, C);
        task(ctx, k, A, B, l, C, X);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);
    // Test ends

    // Clean up
    const auto t2 = std::chrono::steady_clock::now();

    // Read back result
    opencl_check(clEnqueueReadBuffer(ctx.queue, ctx.matrix_x, CL_TRUE, 0, sizeof(float) * M * N,
                                     X.data(), 0, nullptr, nullptr),
                 "read buffer failed");

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
    const auto method{std::string("Parallel OpenCL 32 " + matrix_size)};
    const auto comments{std::string("operation:") + std::string(operation_string)};

    // Output
    {
        const std::string base_file_name =
            "results/parallel_opencl_32_" + matrix_size + "_" + std::string(operation_string);
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = operation_string;
        j["comments"] = comments;
        j["threads"] = 1;
        j["precision"] = "32";
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
