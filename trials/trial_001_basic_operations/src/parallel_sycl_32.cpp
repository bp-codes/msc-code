/**
 * @file parallel_sycl_32.cpp
 * @brief
 *
 * @author Ben Palmer
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026 Ben Palmer
 * SPDX-License-Identifier: MIT
 */

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

#include <helper/Error.hpp>
#include <helper/SyclFunctions.hpp>
#include <helper/helper.hpp>
#include <nlohmann/json.hpp>
#include <sycl/sycl.hpp>

using OperationKind = helper::OperationKind;

namespace {

inline constexpr float MIN_DENOMINATOR{1.0e-9};
inline constexpr std::uint64_t RNG_SEED{123456789ULL};

/**
 * @brief SYCL: c[i] = a[i] + b[i]
 */
sycl::event parallel_add(std::size_t n, sycl::queue& q, const float* sycldev_numbers_a,
                         const float* sycldev_numbers_b, float* sycldev_numbers_c) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const auto idx{std::size_t(i[0])};
        sycldev_numbers_c[idx] = sycldev_numbers_a[idx] + sycldev_numbers_b[idx];
    });
}

/**
 * @brief SYCL: c[i] = a[i] * b[i]
 */
sycl::event parallel_multiply(std::size_t n, sycl::queue& q, const float* sycldev_numbers_a,
                              const float* sycldev_numbers_b, float* sycldev_numbers_c) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const auto idx{std::size_t(i[0])};
        sycldev_numbers_c[idx] = sycldev_numbers_a[idx] * sycldev_numbers_b[idx];
    });
}

/**
 * @brief SYCL: c[i] = a[i] / max(b[i], MIN_DENOMINATOR)
 */
sycl::event parallel_divide(std::size_t n, sycl::queue& q, const float* sycldev_numbers_a,
                            const float* sycldev_numbers_b, float* sycldev_numbers_c) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const auto idx{std::size_t(i[0])};
        const auto denom{SyclFunctions::fmax(sycldev_numbers_b[idx], MIN_DENOMINATOR)};
        sycldev_numbers_c[idx] = sycldev_numbers_a[idx] / denom;
    });
}

/**
 * @brief SYCL: c[i] = pow(a[i], b[i])
 */
sycl::event parallel_power(std::size_t n, sycl::queue& q, const float* sycldev_numbers_a,
                           const float* sycldev_numbers_b, float* sycldev_numbers_c) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const auto idx{std::size_t(i[0])};
        sycldev_numbers_c[idx] = SyclFunctions::pow(sycldev_numbers_a[idx], sycldev_numbers_b[idx]);
    });
}

/**
 * @brief SYCL: c[i] = exp(a[i]) + exp(b[i])
 */
sycl::event parallel_exp(std::size_t n, sycl::queue& q, const float* sycldev_numbers_a,
                         const float* sycldev_numbers_b, float* sycldev_numbers_c) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const auto idx{std::size_t(i[0])};
        sycldev_numbers_c[idx] =
            SyclFunctions::exp(sycldev_numbers_a[idx]) + SyclFunctions::exp(sycldev_numbers_b[idx]);
    });
}

/**
 * @brief SYCL: c[i] = log(a[i]) + log(b[i])
 * @warning Inputs must be > 0. No bounds/validity checking is performed in this hot loop.
 */
sycl::event parallel_log(std::size_t n, sycl::queue& q, const float* sycldev_numbers_a,
                         const float* sycldev_numbers_b, float* sycldev_numbers_c) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const auto idx{std::size_t(i[0])};
        sycldev_numbers_c[idx] =
            SyclFunctions::log(sycldev_numbers_a[idx]) + SyclFunctions::log(sycldev_numbers_b[idx]);
    });
}

/**
 * @brief SYCL: c[i] = sqrt(a[i]) + sqrt(b[i])
 * @warning Inputs must be >= 0. No bounds/validity checking is performed in this hot loop.
 */
sycl::event parallel_sqrt(std::size_t n, sycl::queue& q, const float* sycldev_numbers_a,
                          const float* sycldev_numbers_b, float* sycldev_numbers_c) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const auto idx{std::size_t(i[0])};
        sycldev_numbers_c[idx] = SyclFunctions::sqrt(sycldev_numbers_a[idx]) +
                                 SyclFunctions::sqrt(sycldev_numbers_b[idx]);
    });
}

/**
 * @brief Dispatch the selected operation.
 * @param operation Operation kind.
 * @param n Array size.
 * @param q SYCL queue.
 * @param numbers_a First input vector.
 * @param numbers_b Second input vector.
 * @param numbers_c Output vector (must be pre-sized).
 */
sycl::event parallel_task(OperationKind operation, std::size_t n, sycl::queue& q,
                          const float* sycldev_numbers_a, const float* sycldev_numbers_b,
                          float* sycldev_numbers_c) {
    switch (operation) {
        case OperationKind::Add: {
            return parallel_add(n, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
        }
        case OperationKind::Multiply: {
            return parallel_multiply(n, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
        }
        case OperationKind::Divide: {
            return parallel_divide(n, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
        }
        case OperationKind::Power: {
            return parallel_power(n, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
        }
        case OperationKind::Exp: {
            return parallel_exp(n, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
        }
        case OperationKind::Log: {
            return parallel_log(n, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
        }
        case OperationKind::Sqrt: {
            return parallel_sqrt(n, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
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

        std::string_view device_string = "GPU";
        if (argc >= 5) {
            device_string = argv[4];

            if (device_string != "GPU" && device_string != "CPU") {
                THROW_INVALID_ARGUMENT("device must be GPU or CPU");
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

        auto numbers_a{std::vector<float>{}};
        auto numbers_b{std::vector<float>{}};
        numbers_a.reserve(n);
        numbers_b.reserve(n);

        for (auto i = std::size_t(0); i < n; i++) {
            numbers_a.emplace_back(dist(rng));
            numbers_b.emplace_back(dist(rng));
        }

        // ======= Calculation Starts ========
        const auto t0{std::chrono::steady_clock::now()};

        sycl::queue q = (device_string == "CPU") ? sycl::queue{sycl::cpu_selector_v}
                                                 : sycl::queue{sycl::gpu_selector_v};

        std::cerr << "Using device: " << q.get_device().get_info<sycl::info::device::name>()
                  << "\n";

        // Results
        std::vector<float> numbers_c(n);

        // Allocate device memory once
        float* sycldev_numbers_a = sycl::malloc_device<float>(n, q);
        float* sycldev_numbers_b = sycl::malloc_device<float>(n, q);
        float* sycldev_numbers_c = sycl::malloc_device<float>(n, q);

        // Copy once
        q.memcpy(sycldev_numbers_a, numbers_a.data(), n * sizeof(float)).wait();
        q.memcpy(sycldev_numbers_b, numbers_b.data(), n * sizeof(float)).wait();

        // ======= Start up =======
        const auto t1{std::chrono::steady_clock::now()};
        const auto deadline{t1 + std::chrono::duration<double>(test_time_seconds)};

        auto iters{static_cast<std::uint64_t>(0)};

        sycl::event last;

        // Do as many times as possible before time runs out
        do {
            last = parallel_task(operation, n, q, sycldev_numbers_a, sycldev_numbers_b,
                                 sycldev_numbers_c);
            last.wait();
            iters++;
        } while (std::chrono::steady_clock::now() < deadline);

        last.wait();

        // Copy results back
        q.memcpy(numbers_c.data(), sycldev_numbers_c, n * sizeof(float)).wait();

        // ======= Clean up =======
        const auto t2{std::chrono::steady_clock::now()};

        // Free device allocations
        sycl::free(sycldev_numbers_a, q);
        sycl::free(sycldev_numbers_b, q);
        sycl::free(sycldev_numbers_c, q);

        // ======= Calculation Ends ========

        const auto t3{std::chrono::steady_clock::now()};

        const auto calculated_value{helper::check_sum(numbers_c)};

        const auto time_setup{std::chrono::duration<double>(t1 - t0).count()};
        const auto time_calc{std::chrono::duration<double>(t2 - t1).count()};
        const auto time_cleanup{std::chrono::duration<double>(t3 - t2).count()};
        const auto time_total{std::chrono::duration<double>(t3 - t0).count()};
        const auto time_per_iteration{time_calc / static_cast<double>(iters)};

        const auto method{std::string("Parallel SYCL 32 ") + std::string(device_string)};
        const auto comments{std::string("operation:") + std::string(operation_string)};

        // Output
        {
            const std::string base_file_name =
                "results/parallel_sycl_32_" + std::string(operation_string);
            const std::string json_file =
                base_file_name + "_" + helper::random_suffix(12) + ".json";

            nlohmann::json j;

            // Metadata / identity
            j["file"] = json_file;
            j["method"] = method;
            j["operation"] = operation_string;
            j["comments"] = comments;
            j["threads"] = 1;
            j["precision"] = "32";
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
