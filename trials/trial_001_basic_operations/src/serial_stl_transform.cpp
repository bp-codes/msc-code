/**
 * @file serial_stl.cpp
 * @brief
 *
 * @author Ben Palmer
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026 Ben Palmer
 * SPDX-License-Identifier: MIT
 */

#include <sys/resource.h>

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
#include <vector>

#include "helper/Error.hpp"
#include "helper/helper.hpp"
#include <nlohmann/json.hpp>

using OperationKind = helper::OperationKind;

namespace {

inline constexpr double MIN_DENOMINATOR{1.0e-9};
inline constexpr std::uint64_t RNG_SEED{123456789ULL};

/**
 * @brief Element-wise addition: c[i] = a[i] + b[i]
 */
void serial_add(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b,
                std::vector<double>& numbers_c) {
    std::transform(numbers_a.begin(), numbers_a.end(), numbers_b.begin(), numbers_c.begin(),
                   [](double x, double y) { return x + y; });
}

/**
 * @brief Element-wise multiplication: c[i] = a[i] * b[i]
 */
void serial_multiply(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b,
                     std::vector<double>& numbers_c) {
    std::transform(numbers_a.begin(), numbers_a.end(), numbers_b.begin(), numbers_c.begin(),
                   [](double x, double y) { return x * y; });
}

/**
 * @brief Element-wise division: c[i] = a[i] / max(b[i], MIN_DENOMINATOR)
 */
void serial_divide(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b,
                   std::vector<double>& numbers_c) {
    std::transform(numbers_a.begin(), numbers_a.end(), numbers_b.begin(), numbers_c.begin(),
                   [](double x, double y) { return x / std::fmax(y, MIN_DENOMINATOR); });
}

/**
 * @brief Element-wise power: c[i] = pow(a[i], b[i])
 */
void serial_power(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b,
                  std::vector<double>& numbers_c) {
    std::transform(numbers_a.begin(), numbers_a.end(), numbers_b.begin(), numbers_c.begin(),
                   [](double x, double y) { return std::pow(x, y); });
}

/**
 * @brief Element-wise exp sum: c[i] = exp(a[i]) + exp(b[i])
 */
void serial_exp(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b,
                std::vector<double>& numbers_c) {
    std::transform(numbers_a.begin(), numbers_a.end(), numbers_b.begin(), numbers_c.begin(),
                   [](double x, double y) { return std::exp(x) + std::exp(y); });
}

/**
 * @brief Element-wise log sum: c[i] = log(a[i]) + log(b[i])
 * @warning Inputs must be > 0. No bounds/validity checking is performed in this hot loop.
 */
void serial_log(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b,
                std::vector<double>& numbers_c) {
    std::transform(numbers_a.begin(), numbers_a.end(), numbers_b.begin(), numbers_c.begin(),
                   [](double x, double y) { return std::log(x) + std::log(y); });
}

/**
 * @brief Element-wise sqrt sum: c[i] = sqrt(a[i]) + sqrt(b[i])
 * @warning Inputs must be >= 0. No bounds/validity checking is performed in this hot loop.
 */
void serial_sqrt(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b,
                 std::vector<double>& numbers_c) {
    std::transform(numbers_a.begin(), numbers_a.end(), numbers_b.begin(), numbers_c.begin(),
                   [](double x, double y) { return std::sqrt(x) + std::sqrt(y); });
}

/**
 * @brief Dispatch the selected operation.
 * @param operation Operation kind.
 * @param numbers_a First input vector.
 * @param numbers_b Second input vector.
 * @param numbers_c Output vector (must be pre-sized).
 */
void serial_task(OperationKind operation, const std::vector<double>& numbers_a,
                 const std::vector<double>& numbers_b, std::vector<double>& numbers_c) {
    switch (operation) {
        case OperationKind::Add: {
            serial_add(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Multiply: {
            serial_multiply(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Divide: {
            serial_divide(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Power: {
            serial_power(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Exp: {
            serial_exp(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Log: {
            serial_log(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Sqrt: {
            serial_sqrt(numbers_a, numbers_b, numbers_c);
            return;
        }
    }

    THROW_RUNTIME_ERROR("Unhandled OperationKind value.");
}

}  // namespace

/**
 * @brief Entry point into program.
 */
auto main(int argc, char** argv) -> int {
    try {
        if (argc < 4) {
            THROW_INVALID_ARGUMENT("Usage: serial.x time_limit vec_size operation");
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

        const auto t1{std::chrono::steady_clock::now()};
        const auto deadline{t1 + std::chrono::duration<double>(test_time_seconds)};

        auto iters{static_cast<std::uint64_t>(0)};
        auto numbers_c{std::vector<double>(n)};
        helper::validate_sizes(numbers_a, numbers_b, numbers_c);

        do {
            serial_task(operation, numbers_a, numbers_b, numbers_c);
            iters++;
        } while (std::chrono::steady_clock::now() < deadline);

        const auto t2{std::chrono::steady_clock::now()};
        const auto t3{std::chrono::steady_clock::now()};

        // ======= Calculation Ends ========

        const auto calculated_value{helper::check_sum(numbers_c)};

        const auto time_setup{std::chrono::duration<double>(t1 - t0).count()};
        const auto time_calc{std::chrono::duration<double>(t2 - t1).count()};
        const auto time_cleanup{std::chrono::duration<double>(t3 - t2).count()};
        const auto time_total{std::chrono::duration<double>(t3 - t0).count()};
        const auto time_per_iteration{time_calc / static_cast<double>(iters)};

        const auto method{std::string("Serial STL Transform")};
        const auto comments{std::string("operation:") + std::string(operation_string)};

        // Output
        {
            const std::string base_file_name =
                "results/serial_stl_transform_" + std::string(operation_string);
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
