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

#include <quadmath.h>

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
#include "helper/helper.hpp"

#include <nlohmann/json.hpp>

// Serial task - sum numbers in the vector
[[nodiscard]]
double task(const std::vector<double>& numbers) {
    __float128 sum{0.0Q};
    for (const auto val : numbers) {
        sum += static_cast<__float128>(val);
    }
    return static_cast<double>(sum);
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

    // Setup
    auto t0 = std::chrono::steady_clock::now();

    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time_seconds);
    std::uint64_t iters = 0;

    double expected_value{};

    // Do as many times as possible before time runs out
    do {
        expected_value = task(numbers);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);

    // Clean up
    auto t2 = std::chrono::steady_clock::now();

    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    auto time_setup = std::chrono::duration<double>(t1 - t0).count();
    auto time_calc = std::chrono::duration<double>(t2 - t1).count();
    auto time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    auto time_total = std::chrono::duration<double>(t3 - t0).count();
    auto time_per_iteration = time_calc / iters;

    // Output
    {
        const auto method{std::string("Precise")};
        const auto operation_string = std::string("sum");
        const auto comments{std::string("operation:") + std::string(operation_string)};

        const std::string base_file_name = "results/precise";
        const std::string json_file = base_file_name + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = operation_string;
        j["comments"] = comments;
        j["threads"] = 1;
        j["precision"] = "128";
        j["device"] = "CPU";

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
