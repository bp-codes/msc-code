/**
 * @file parallel_cuda.cpp
 * @brief
 *
 * @author Ben Palmer
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026 Ben Palmer
 * SPDX-License-Identifier: MIT
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "helper/Error.hpp"
#include "helper/helper.hpp"
#include <nlohmann/json.hpp>

using OperationKind = helper::OperationKind;

namespace {

inline constexpr float MIN_DENOMINATOR{1.0e-9f};
inline constexpr std::uint64_t RNG_SEED{123456789ULL};

inline void cuda_check(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        (void)status;
        THROW_RUNTIME_ERROR(message);
    }
}

__global__ void kernel_add(std::size_t n, const float* a, const float* b, float* c) {
    const auto idx{std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void kernel_multiply(std::size_t n, const float* a, const float* b, float* c) {
    const auto idx{std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void kernel_divide(std::size_t n, const float* a, const float* b, float* c) {
    const auto idx{std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n) {
        const auto denom{b[idx] > MIN_DENOMINATOR ? b[idx] : MIN_DENOMINATOR};
        c[idx] = a[idx] / denom;
    }
}

__global__ void kernel_power(std::size_t n, const float* a, const float* b, float* c) {
    const auto idx{std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n) {
        c[idx] = pow(a[idx], b[idx]);
    }
}

__global__ void kernel_exp(std::size_t n, const float* a, const float* b, float* c) {
    const auto idx{std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n) {
        c[idx] = exp(a[idx]) + exp(b[idx]);
    }
}

__global__ void kernel_log(std::size_t n, const float* a, const float* b, float* c) {
    const auto idx{std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n) {
        c[idx] = log(a[idx]) + log(b[idx]);
    }
}

__global__ void kernel_sqrt(std::size_t n, const float* a, const float* b, float* c) {
    const auto idx{std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n) {
        c[idx] = sqrt(a[idx]) + sqrt(b[idx]);
    }
}

cudaError_t launch_kernel(OperationKind operation, std::size_t n, cudaStream_t stream,
                          const float* a, const float* b, float* c) {
    constexpr int BLOCK_SIZE{256};
    const auto grid_size{
        static_cast<unsigned int>((n + std::size_t(BLOCK_SIZE) - 1) / std::size_t(BLOCK_SIZE))};

    switch (operation) {
        case OperationKind::Add: {
            kernel_add<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Multiply: {
            kernel_multiply<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Divide: {
            kernel_divide<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Power: {
            kernel_power<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Exp: {
            kernel_exp<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Log: {
            kernel_log<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Sqrt: {
            kernel_sqrt<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
    }

    return cudaErrorInvalidValue;
}

}  // namespace

/**
 * @brief Entry point into program.
 */
auto main(int argc, char** argv) -> int {
    try {
        if (argc < 4) {
            THROW_INVALID_ARGUMENT("Usage: cuda.x time_limit vec_size operation");
        }

        const auto test_time_seconds{helper::parse_floating_point(argv[1])};
        const auto n{helper::parse_size(argv[2])};
        const auto operation_string{std::string_view(argv[3])};
        const auto operation{helper::parse_operation(operation_string)};

        if (test_time_seconds <= 0.0) {
            THROW_INVALID_ARGUMENT("time_limit must be > 0.");
        }
        if (n == 0) {
            THROW_INVALID_ARGUMENT("vec_size must be > 0.");
        }

        std::mt19937_64 rng(RNG_SEED);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        auto numbers_a{std::vector<float>{}};
        auto numbers_b{std::vector<float>{}};
        numbers_a.reserve(n);
        numbers_b.reserve(n);

        for (auto i = std::size_t(0); i < n; i++) {
            numbers_a.emplace_back(static_cast<float>(dist(rng)));
            numbers_b.emplace_back(static_cast<float>(dist(rng)));
        }

        // ======= Calculation Starts ========
        const auto t0{std::chrono::steady_clock::now()};

        int device_id{0};
        cuda_check(cudaGetDevice(&device_id), "cudaGetDevice failed.");

        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties failed.");
        std::cerr << "Using device: " << prop.name << "\n";

        // Results
        auto numbers_c{std::vector<float>(n)};

        // Allocate device memory once
        float* dev_a{};
        float* dev_b{};
        float* dev_c{};

        cuda_check(cudaMalloc(&dev_a, n * sizeof(float)), "cudaMalloc dev_a failed.");
        cuda_check(cudaMalloc(&dev_b, n * sizeof(float)), "cudaMalloc dev_b failed.");
        cuda_check(cudaMalloc(&dev_c, n * sizeof(float)), "cudaMalloc dev_c failed.");

        cudaStream_t stream{};
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate failed.");

        // Copy once
        cuda_check(cudaMemcpyAsync(dev_a, numbers_a.data(), n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync dev_a failed.");
        cuda_check(cudaMemcpyAsync(dev_b, numbers_b.data(), n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync dev_b failed.");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize after H2D copies failed.");

        // ======= Start up =======
        const auto t1{std::chrono::steady_clock::now()};
        const auto deadline{t1 + std::chrono::duration<double>(test_time_seconds)};

        auto iters{static_cast<std::uint64_t>(0)};

        // Do as many times as possible before time runs out
        do {
            cuda_check(launch_kernel(operation, n, stream, dev_a, dev_b, dev_c),
                       "Kernel launch failed.");
            iters++;
        } while (std::chrono::steady_clock::now() < deadline);

        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize after kernels failed.");

        // Copy results back
        cuda_check(cudaMemcpyAsync(numbers_c.data(), dev_c, n * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync D2H failed.");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize after D2H copy failed.");

        // ======= Clean up =======
        const auto t2{std::chrono::steady_clock::now()};

        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy failed.");
        cuda_check(cudaFree(dev_a), "cudaFree dev_a failed.");
        cuda_check(cudaFree(dev_b), "cudaFree dev_b failed.");
        cuda_check(cudaFree(dev_c), "cudaFree dev_c failed.");

        // ======= Calculation Ends ========
        const auto t3{std::chrono::steady_clock::now()};

        const auto calculated_value{helper::check_sum(numbers_c)};

        const auto time_setup{std::chrono::duration<double>(t1 - t0).count()};
        const auto time_calc{std::chrono::duration<double>(t2 - t1).count()};
        const auto time_cleanup{std::chrono::duration<double>(t3 - t2).count()};
        const auto time_total{std::chrono::duration<double>(t3 - t0).count()};
        const auto time_per_iteration{time_calc / static_cast<double>(iters)};

        const auto method{std::string("Parallel CUDA")};
        const auto comments{std::string("operation:") + std::string(operation_string)};

        // Output
        {
            const std::string base_file_name =
                "results/parallel_cuda_" + std::string(operation_string);
            const std::string json_file =
                base_file_name + "_" + helper::random_suffix(12) + ".json";

            nlohmann::json j;

            // Metadata / identity
            j["file"] = json_file;
            j["method"] = method;
            j["operation"] = operation_string;
            j["comments"] = comments;
            j["threads"] = 1;
            j["precision"] = "64";
            j["device"] = "GPU";

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
            j["values"] = helper::to_string_precise_vector(numbers_c);

            // Memory
            // j["max_rss_kb"] = max_rss_kb();

            std::ofstream out(json_file);
            if (!out) {
                throw std::runtime_error("Failed to open output JSON file.");
            }

            // Save JSON file.
            out << std::setw(2) << j << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
