// serial.cpp
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "Error.hpp"
#include "helper.hpp"
#include "json.hpp"

std::size_t floor_pow2(std::size_t x) {
    std::size_t p = 1;
    while ((p << 1) <= x) {
        p <<= 1;
    }
    return p;
}

float task(const std::vector<float>& numbers) {
    auto max_threads = omp_get_max_threads();  // hardware / env limit
    auto threads = floor_pow2(max_threads);    // force power-of-two threads

    std::vector<float> partial(threads, 0.0f);
    auto result{0.0f};
    const std::size_t numbers_size = numbers.size();

#pragma omp parallel num_threads(threads)
    {
        auto tid = omp_get_thread_num();

        // 1) local sum
        auto local{0.0};
#pragma omp for schedule(static)
        for (auto i = std::size_t{0}; i < numbers_size; ++i) {
            local += numbers[i];
        }
        partial[tid] = local;

#pragma omp barrier

        // 2) tree reduction
        for (auto offset = threads >> std::size_t{1}; offset > std::size_t{0};
             offset >>= std::size_t{1}) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
#pragma omp barrier
        }

// 3) final result in thread 0
#pragma omp single
        result = partial[0];
    }

    return result;
}

// Serial task - sum numbers in the vector
float serial_naive_task(const std::vector<float>& numbers) {
    auto sum{0.0f};
    for (const auto val : numbers) {
        sum += val;
    }
    return sum;
}

int main(int argc, char** argv) {
    // Set threads
    omp_set_num_threads(helper::get_num_threads());

    // Must have 3 arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }

    // Read in test_time and size of vector
    double test_time_seconds = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);
    const std::string operation = "Sum vector elements.";

    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);  // [0.0, 1.0)

    // Vector of numbers
    std::vector<float> numbers;
    numbers.reserve(N);

    // Populate vector
    for (int i = 0; i < N; ++i) {
        numbers.emplace_back(static_cast<float>(dist(rng)));
    }

    auto expected_value = serial_naive_task(numbers);

    // ======= Calculation Starts ========

    // Setup
    auto t0 = std::chrono::steady_clock::now();

    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time_seconds);
    std::uint64_t iters = 0;

    float calculated_value{};

    // Do as many times as possible before time runs out
    do {
        calculated_value = task(numbers);
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

    bool passed_check = std::abs(calculated_value - expected_value) < 1.0e-9;

    // Output
    {
        const auto method{std::string("Parallel OpenMP Tree 32")};
        const auto operation_string = std::string("sum");
        const auto comments{std::string("operation:") + std::string(operation_string)};

        const std::string base_file_name = "results/parallel_openmp_tree_32_" + operation_string;
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = operation_string;
        j["comments"] = comments;
        j["threads"] = 1;
        j["precision"] = "32";
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
        j["calculated_value"] = helper::to_string_precise(calculated_value);
        ;
        j["difference"] = helper::to_string_precise(expected_value - calculated_value);
        j["passed_check"] = passed_check;
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
