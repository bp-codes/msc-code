// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <limits>
#include <omp.h>
#include <random>


static inline float stopping_power_fe(const float v_m_per_s) 
{
    static constexpr float c     = 299792458.0f;
    static constexpr float me    = 0.51099895000f;
    static constexpr float K     = 0.307075f;

    static constexpr float Z     = 26.0f;
    static constexpr float A     = 55.845f;
    static constexpr float rho   = 7.874f;
    static constexpr float I     = 286.0e-6f;

    static constexpr float M     = 938.2720813f;

    const float beta  = v_m_per_s / c;
    const float beta2 = beta * beta;
    const float inv_1_minus_beta2 = 1.0f / (1.0f - beta2);
    const float gamma2 = inv_1_minus_beta2;              // gamma^2
    const float gamma  = std::sqrt(gamma2);

    const float me_over_M = me / M;

    const float numerator = 2.0f * me * beta2 * gamma2;
    const float denom = 1.0f + 2.0f * gamma * me_over_M + (me_over_M * me_over_M);
    const float Tmax = numerator / denom;

    const float logArg = (2.0f * me * beta2 * gamma2 * Tmax) / (I * I);

    const float prefactor = K * (Z / A) / beta2;
    const float bracket   = 0.5f * std::log(logArg) - beta2;

    return (prefactor * bracket) * rho;
}



void serial_task(const std::vector<float>& velocity_array, std::vector<float>& results)
{
    for(size_t i=0; i<velocity_array.size(); i++)
    {
        results[i] = stopping_power_fe(velocity_array[i]);
    }
}



// Check the sum of an array (serial)
float check_sum(const std::vector<float>& numbers_c)
{
    float sum {};
    for(const auto& number : numbers_c)
    {
        sum = sum + number;
    }
    return sum;
}


int main(int argc, char** argv) 
{

    // Must have 3 arguments
    if (argc < 3) 
    {
        std::cerr << "Usage: " << argv[0] << " time_limit   vec_size\n";
        return 1;
    }

    // Read in test_time and size of vector
    double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);


    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);    // [0.0, 1.0)

    // Vector of numbers
    std::vector<float> velocity_array;
    velocity_array.reserve(N);

    // Populate vectors
    for (std::size_t i = 0; i < N; ++i) 
    {
        velocity_array.emplace_back(1.0e6f * dist(rng));
    }

    float expected_value {};

    // Expected value
    {
        std::vector<float> stopping_power(N);
        serial_task(velocity_array, stopping_power);
        expected_value = check_sum(stopping_power);
        std::cout << "Serial computed expected value:  " << expected_value << std::endl;
    }

  
    // ======= Calculation Starts ========

    // Setup
    auto t0 = std::chrono::steady_clock::now();


    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    int sum {};

    
    // Do as many times as possible before time runs out
    std::vector<float> stopping_power(N);
    do 
    {
        serial_task(velocity_array, stopping_power);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);

    // Clean up
    auto t2 = std::chrono::steady_clock::now();


    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    // Check
    float calculated_value = check_sum(stopping_power);
   
    double time_setup = std::chrono::duration<double>(t1 - t0).count();
    double time_calc = std::chrono::duration<double>(t2 - t1).count();
    double time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    double time_total = std::chrono::duration<double>(t3 - t0).count();
    double time_per_iteration = time_calc / iters;


    std::string method {"Serial"};
    std::string comments {"stopping_power"};
    bool passed_check = std::abs(calculated_value - expected_value) < 1.0e-6f;

    std::cout << method << "," 
              << expected_value << "," 
              << calculated_value << "," 
              << iters << "," 
              << time_per_iteration << "," 
              << time_setup << "," 
              << time_calc << "," 
              << time_cleanup << "," 
              << time_total << "," 
              << passed_check << "," 
              << comments << "" 
              << std::endl;

    
    
    return 0;
}
