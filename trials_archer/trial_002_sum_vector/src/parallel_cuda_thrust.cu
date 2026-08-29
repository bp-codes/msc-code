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

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "helper/Error.hpp"
#include "helper/helper_cuda.hpp"

#include <nlohmann/json.hpp>

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err = call;                                                                 \
        if (err != cudaSuccess) {                                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                 \
            std::exit(EXIT_FAILURE);                                                            \
        }                                                                                       \
    } while (0)

double task(const double* d_input, int N) {
    thrust::device_ptr<const double> begin(d_input);
    thrust::device_ptr<const double> end = begin + N;

    return thrust::reduce(thrust::device, begin, end, 0.0, thrust::plus<double>());
}

int main(int argc, char** argv) {
    // Must have 3 arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }

    // Read in test_time and size of vector
    const double test_time_seconds = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);

    if (N <= 0) {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }

    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);  // [0.0, 1.0)

    // Vector of numbers
    std::vector<double> numbers;
    numbers.reserve(N);

    // Populate vector
    for (int i = 0; i < N; ++i) {
        numbers.emplace_back(dist(rng));
    }

    // ======= Calculation Starts ========

    auto t0 = std::chrono::steady_clock::now();

    // Allocate on device
    double* device_numbers = nullptr;
    CUDA_CHECK(cudaMalloc(&device_numbers, N * sizeof(double)));

    // Copy input to device
    CUDA_CHECK(
        cudaMemcpy(device_numbers, numbers.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time_seconds);
    std::uint64_t iters = 0;

    double calculated_value{};

    do {
        calculated_value = task(device_numbers, N);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);

    // Clean up
    auto t2 = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaFree(device_numbers));

    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    auto time_setup = std::chrono::duration<double>(t1 - t0).count();
    auto time_calc = std::chrono::duration<double>(t2 - t1).count();
    auto time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    auto time_total = std::chrono::duration<double>(t3 - t0).count();
    auto time_per_iteration = time_calc / iters;

    std::string method{"CUDA"};
    std::string device{"gpu"};

    // Output
    {
        const auto method{std::string("Parallel CUDA Thrust")};
        const auto operation_string = std::string("sum");
        const auto comments{std::string("operation:") + std::string(operation_string)};

        const std::string base_file_name = "results/parallel_cuda_thrust_" + operation_string;
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
        j["values"] = helper::to_string_precise_vector(numbers);

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
