// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <numeric>
#include <algorithm>

#include "Error.hpp"
#include "helper.hpp"
#include "json.hpp"

#include <thread>
#include <immintrin.h>


// Parallel task - sum numbers in the vector
[[nodiscard]]
float task(const std::vector<float>& numbers)
{
    const auto n {std::size_t(numbers.size())};

    if (n == 0)
    {
        return 0.0f;
    }

    const std::size_t num_threads {std::min(helper::get_num_threads(), n)};
    const std::size_t chunk {(n + num_threads - 1) / num_threads};

    struct alignas(64) PaddedSum
    {
        float value {0.0f};
    };

    std::vector<std::thread> threads;
    std::vector<PaddedSum> reduction_sum(num_threads);
    threads.reserve(num_threads);

    for (std::size_t t = 0; t < num_threads; ++t)
    {
        const std::size_t begin {t * chunk};
        const std::size_t end {std::min(begin + chunk, n)};

        if (begin >= n)
        {
            break;
        }

        threads.emplace_back(
            [&, t, begin, end]()
            {
                constexpr std::size_t simd_width {8}; // 8 floats in AVX

                __m256 vsum {_mm256_setzero_ps()};
                std::size_t i {begin};

                for (; i + simd_width <= end; i += simd_width)
                {
                    const __m256 v {_mm256_loadu_ps(&numbers[i])};
                    vsum = _mm256_add_ps(vsum, v);
                }

                alignas(32) float temp[8];
                _mm256_store_ps(temp, vsum);

                float local_sum {
                    temp[0] + temp[1] + temp[2] + temp[3] +
                    temp[4] + temp[5] + temp[6] + temp[7]
                };

                for (; i < end; ++i)
                {
                    local_sum += numbers[i];
                }

                reduction_sum[t].value = local_sum;
            }
        );
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    float sum {0.0f};
    for (std::size_t t = 0; t < threads.size(); ++t)
    {
        sum += reduction_sum[t].value;
    }

    return sum;
}



// Serial task - sum numbers in the vector
float serial_naive_task(const std::vector<float>& numbers)
{
    auto sum {0.0f};
    for(const auto val : numbers)
    {
        sum += val;
    }
    return sum;
}



int main(int argc, char** argv) 
{

    // Must have 3 arguments
    if (argc < 3) 
    {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }
    
    // Read in test_time and size of vector
    const double test_time_seconds = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);
    const std::string operation = "Sum vector elements.";

    if(N <= 0)
    {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }
    
    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);  // [0.0, 1.0)

    // Vector of numbers
    std::vector<float> numbers;
    numbers.reserve(N);

    // Populate vector
    for (int i = 0; i < N; ++i) 
    {
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

    float calculated_value {};

    // Do as many times as possible before time runs out
    do 
    {
        calculated_value = task(numbers);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);

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
        const auto method {std::string("Parallel Thread SIMD")};
        const auto operation_string = std::string("sum");
        const auto comments {std::string("operation:") + std::string(operation_string)};

        const std::string base_file_name = "results/parallel_thread_simd_" + operation_string;
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = operation_string;
        j["comments"] = comments;
        j["threads"] = helper::get_num_threads();
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
        j["calculated_value"] = helper::to_string_precise(calculated_value);;
        j["difference"] = helper::to_string_precise(expected_value - calculated_value);
        j["passed_check"] = passed_check;
        j["values"] = helper::to_string_precise_vector(numbers);

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
