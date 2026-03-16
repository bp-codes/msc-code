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
#include <algorithm>

#include "Error.hpp"
#include "helper.hpp"
#include "json.hpp"

#include <immintrin.h>
#include <cstddef>


float task(const std::vector<float>& numbers)
{
    const float* p = numbers.data();
    std::size_t n = numbers.size();

    __m256 vsum = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(p + i);
        vsum = _mm256_add_ps(vsum, v);
    }

    // horizontal sum of vsum (8 floats -> 1 float)
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);

    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);

    float sum = _mm_cvtss_f32(sum1);

    // tail
    for (; i < n; ++i)
    {
        sum += p[i];
    }

    return sum;
}


// Serial task - sum numbers in the vector
double serial_naive_task(const std::vector<float>& numbers)
{
    auto sum {0.0};
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

    double calculated_value_float {};

    // Do as many times as possible before time runs out
    do 
    {
        calculated_value_float = task(numbers);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);

    auto calculated_value = static_cast<float>(calculated_value_float);

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
        const auto method {std::string("Serial SIMD 32")};
        const auto operation_string = std::string("sum");
        const auto comments {std::string("operation:") + std::string(operation_string)};

        const std::string base_file_name = "results/serial_simd_32_" + operation_string;
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = operation_string;
        j["comments"] = comments;
        j["threads"] = 1;

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
