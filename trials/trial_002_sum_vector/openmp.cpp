// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>


// Parallel task - sum numbers in the vector
double task(const std::vector<double>& numbers)
{
    auto sum {0.0};
    const std::size_t numbers_size = numbers.size();

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (auto i = std::size_t{0}; i < numbers_size; i++)
    {
        sum += numbers[i];
    }
    return sum;
}


// Serial task - sum numbers in the vector
double serial_task(const std::vector<double>& numbers)
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
    const double test_time = std::atof(argv[1]);
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
    std::vector<double> numbers;
    numbers.reserve(N);

    // Populate vector
    for (int i = 0; i < N; ++i) 
    {
        numbers.emplace_back(dist(rng));
    }

    auto expected_value = serial_task(numbers);
    

    // ======= Calculation Starts ========

    // Setup
    auto t0 = std::chrono::steady_clock::now();

    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    double calculated_value {};

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

    std::string method {"OpenMP"};
    std::string comments {"operation:" + operation};
    bool passed_check = std::abs(calculated_value - expected_value) < 1.0e-9;

   
    // Technology,Iterations,Sum,TimePerIteration,SetupTime
    std::cout << method << "," 
              << std::scientific << std::setprecision(9)
              << expected_value << "," 
              << calculated_value << "," 
              << std::scientific << std::setprecision(6)
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
