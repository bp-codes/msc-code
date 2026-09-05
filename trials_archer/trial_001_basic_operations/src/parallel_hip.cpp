/**
 * @file parallel_hip.cpp
 * @brief HIP version of the simple vector operation benchmark.
 *
 * @author Ben Palmer
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026 Ben Palmer
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <helper/Error.hpp>
#include <helper/helper.hpp>
#include <nlohmann/json.hpp>

using OperationKind = helper::OperationKind;

namespace {

inline constexpr double MIN_DENOMINATOR{1.0e-9};
inline constexpr std::uint64_t RNG_SEED{123456789ULL};
inline constexpr int HIP_BLOCK_SIZE{256};

/**
 * @brief Throw if a HIP runtime call fails.
 */
void hip_check(const hipError_t err, const char* call, const char* file, const int line) {
    if (err != hipSuccess) {
        std::ostringstream oss;
        oss << "HIP error from " << call << " at " << file << ':' << line << ": "
            << hipGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

#define HIP_CHECK(call) hip_check((call), #call, __FILE__, __LINE__)

/**
 * @brief HIP: c[i] = a[i] + b[i]
 */
__global__ void hip_add(std::size_t n, const double* hipdev_numbers_a,
                        const double* hipdev_numbers_b, double* hipdev_numbers_c) {
    const auto idx{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};

    if (idx < n) {
        hipdev_numbers_c[idx] = hipdev_numbers_a[idx] + hipdev_numbers_b[idx];
    }
}

/**
 * @brief HIP: c[i] = a[i] * b[i]
 */
__global__ void hip_multiply(std::size_t n, const double* hipdev_numbers_a,
                             const double* hipdev_numbers_b, double* hipdev_numbers_c) {
    const auto idx{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};

    if (idx < n) {
        hipdev_numbers_c[idx] = hipdev_numbers_a[idx] * hipdev_numbers_b[idx];
    }
}

/**
 * @brief HIP: c[i] = a[i] / max(b[i], MIN_DENOMINATOR)
 */
__global__ void hip_divide(std::size_t n, const double* hipdev_numbers_a,
                           const double* hipdev_numbers_b, double* hipdev_numbers_c) {
    const auto idx{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};

    if (idx < n) {
        const auto denom{fmax(hipdev_numbers_b[idx], MIN_DENOMINATOR)};
        hipdev_numbers_c[idx] = hipdev_numbers_a[idx] / denom;
    }
}

/**
 * @brief HIP: c[i] = pow(a[i], b[i])
 */
__global__ void hip_power(std::size_t n, const double* hipdev_numbers_a,
                          const double* hipdev_numbers_b, double* hipdev_numbers_c) {
    const auto idx{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};

    if (idx < n) {
        hipdev_numbers_c[idx] = pow(hipdev_numbers_a[idx], hipdev_numbers_b[idx]);
    }
}

/**
 * @brief HIP: c[i] = exp(a[i]) + exp(b[i])
 */
__global__ void hip_exp(std::size_t n, const double* hipdev_numbers_a,
                        const double* hipdev_numbers_b, double* hipdev_numbers_c) {
    const auto idx{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};

    if (idx < n) {
        hipdev_numbers_c[idx] = exp(hipdev_numbers_a[idx]) + exp(hipdev_numbers_b[idx]);
    }
}

/**
 * @brief HIP: c[i] = log(a[i]) + log(b[i])
 * @warning Inputs must be > 0. No bounds/validity checking is performed in this hot loop.
 */
__global__ void hip_log(std::size_t n, const double* hipdev_numbers_a,
                        const double* hipdev_numbers_b, double* hipdev_numbers_c) {
    const auto idx{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};

    if (idx < n) {
        hipdev_numbers_c[idx] = log(hipdev_numbers_a[idx]) + log(hipdev_numbers_b[idx]);
    }
}

/**
 * @brief HIP: c[i] = sqrt(a[i]) + sqrt(b[i])
 * @warning Inputs must be >= 0. No bounds/validity checking is performed in this hot loop.
 */
__global__ void hip_sqrt(std::size_t n, const double* hipdev_numbers_a,
                         const double* hipdev_numbers_b, double* hipdev_numbers_c) {
    const auto idx{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};

    if (idx < n) {
        hipdev_numbers_c[idx] = sqrt(hipdev_numbers_a[idx]) + sqrt(hipdev_numbers_b[idx]);
    }
}

/**
 * @brief Dispatch the selected HIP operation.
 * @param operation Operation kind.
 * @param n Array size.
 * @param hipdev_numbers_a First input vector on device.
 * @param hipdev_numbers_b Second input vector on device.
 * @param hipdev_numbers_c Output vector on device.
 */
void parallel_task(OperationKind operation, std::size_t n, const double* hipdev_numbers_a,
                   const double* hipdev_numbers_b, double* hipdev_numbers_c) {
    const auto blocks{static_cast<unsigned int>((n + HIP_BLOCK_SIZE - 1) / HIP_BLOCK_SIZE)};
    const dim3 grid{blocks};
    const dim3 block{HIP_BLOCK_SIZE};

    switch (operation) {
        case OperationKind::Add: {
            hipLaunchKernelGGL(hip_add, grid, block, 0, 0, n, hipdev_numbers_a, hipdev_numbers_b,
                               hipdev_numbers_c);
            break;
        }
        case OperationKind::Multiply: {
            hipLaunchKernelGGL(hip_multiply, grid, block, 0, 0, n, hipdev_numbers_a,
                               hipdev_numbers_b, hipdev_numbers_c);
            break;
        }
        case OperationKind::Divide: {
            hipLaunchKernelGGL(hip_divide, grid, block, 0, 0, n, hipdev_numbers_a,
                               hipdev_numbers_b, hipdev_numbers_c);
            break;
        }
        case OperationKind::Power: {
            hipLaunchKernelGGL(hip_power, grid, block, 0, 0, n, hipdev_numbers_a,
                               hipdev_numbers_b, hipdev_numbers_c);
            break;
        }
        case OperationKind::Exp: {
            hipLaunchKernelGGL(hip_exp, grid, block, 0, 0, n, hipdev_numbers_a, hipdev_numbers_b,
                               hipdev_numbers_c);
            break;
        }
        case OperationKind::Log: {
            hipLaunchKernelGGL(hip_log, grid, block, 0, 0, n, hipdev_numbers_a, hipdev_numbers_b,
                               hipdev_numbers_c);
            break;
        }
        case OperationKind::Sqrt: {
            hipLaunchKernelGGL(hip_sqrt, grid, block, 0, 0, n, hipdev_numbers_a,
                               hipdev_numbers_b, hipdev_numbers_c);
            break;
        }
        default: {
            THROW_RUNTIME_ERROR("Unhandled OperationKind value.");
        }
    }

    HIP_CHECK(hipGetLastError());
}

}  // namespace

/**
 * @brief Entry point into program.
 */
auto main(int argc, char** argv) -> int {
    try {
        if (argc < 4) {
            THROW_INVALID_ARGUMENT("Usage: parallel_hip.x time_limit vec_size operation [GPU]");
        }

        const auto test_time_seconds{helper::parse_floating_point(argv[1])};
        const auto n{helper::parse_size(argv[2])};
        const auto operation_string{std::string_view(argv[3])};
        const auto operation{helper::parse_operation(operation_string)};

        std::string_view device_string = "GPU";
        if (argc >= 5) {
            device_string = argv[4];

            if (device_string != "GPU") {
                THROW_INVALID_ARGUMENT("HIP version only supports device GPU");
            }
        }

        if (test_time_seconds <= 0.0) {
            THROW_INVALID_ARGUMENT("time_limit must be > 0.");
        }
        if (n == 0) {
            THROW_INVALID_ARGUMENT("vec_size must be > 0.");
        }

        std::mt19937_64 rng(RNG_SEED);
        std::uniform_real_distribution<double> dist(1.0, 2.0);

        auto numbers_a{std::vector<double>{}};
        auto numbers_b{std::vector<double>{}};
        numbers_a.reserve(n);
        numbers_b.reserve(n);

        for (auto i = std::size_t(0); i < n; i++) {
            numbers_a.emplace_back(dist(rng));
            numbers_b.emplace_back(dist(rng));
        }

        // ======= Calculation Starts ========
        const auto t0{std::chrono::steady_clock::now()};

        int device_count{};
        HIP_CHECK(hipGetDeviceCount(&device_count));
        if (device_count <= 0) {
            THROW_RUNTIME_ERROR("No HIP devices found.");
        }

        HIP_CHECK(hipSetDevice(0));

        hipDeviceProp_t props{};
        HIP_CHECK(hipGetDeviceProperties(&props, 0));
        std::cerr << "Using device: " << props.name << "\n";

        // Results
        std::vector<double> numbers_c(n);

        // Allocate device memory once
        double* hipdev_numbers_a{};
        double* hipdev_numbers_b{};
        double* hipdev_numbers_c{};

        HIP_CHECK(hipMalloc(&hipdev_numbers_a, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&hipdev_numbers_b, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&hipdev_numbers_c, n * sizeof(double)));

        // Copy once
        HIP_CHECK(hipMemcpy(hipdev_numbers_a, numbers_a.data(), n * sizeof(double),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(hipdev_numbers_b, numbers_b.data(), n * sizeof(double),
                            hipMemcpyHostToDevice));

        // ======= Start up =======
        const auto t1{std::chrono::steady_clock::now()};
        const auto deadline{t1 + std::chrono::duration<double>(test_time_seconds)};

        auto iters{static_cast<std::uint64_t>(0)};

        // Do as many times as possible before time runs out
        do {
            parallel_task(operation, n, hipdev_numbers_a, hipdev_numbers_b, hipdev_numbers_c);
            HIP_CHECK(hipDeviceSynchronize());
            iters++;
        } while (std::chrono::steady_clock::now() < deadline);

        HIP_CHECK(hipDeviceSynchronize());

        // Copy results back
        HIP_CHECK(hipMemcpy(numbers_c.data(), hipdev_numbers_c, n * sizeof(double),
                            hipMemcpyDeviceToHost));

        // ======= Clean up =======
        const auto t2{std::chrono::steady_clock::now()};

        // Free device allocations
        HIP_CHECK(hipFree(hipdev_numbers_a));
        HIP_CHECK(hipFree(hipdev_numbers_b));
        HIP_CHECK(hipFree(hipdev_numbers_c));

        // ======= Calculation Ends ========
        const auto t3{std::chrono::steady_clock::now()};

        const auto calculated_value{helper::check_sum(numbers_c)};

        const auto time_setup{std::chrono::duration<double>(t1 - t0).count()};
        const auto time_calc{std::chrono::duration<double>(t2 - t1).count()};
        const auto time_cleanup{std::chrono::duration<double>(t3 - t2).count()};
        const auto time_total{std::chrono::duration<double>(t3 - t0).count()};
        const auto time_per_iteration{time_calc / static_cast<double>(iters)};

        const auto method{std::string("Parallel HIP ") + std::string(device_string)};
        const auto comments{std::string("operation:") + std::string(operation_string)};

        // Output
        {
            const std::string base_file_name =
                "results/parallel_hip_" + std::string(operation_string);
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
            j["device"] = device_string;

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
            j["max_rss_kb"] = helper::max_rss_kb();

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
