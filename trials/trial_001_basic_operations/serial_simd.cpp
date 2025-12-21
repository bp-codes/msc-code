// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <algorithm>




//======================================
// Serial SIMD
//======================================

void serial_simd_add(const std::vector<double>& a,
              const std::vector<double>& b,
              std::vector<double>& c)
{
    size_t n = a.size();
    size_t i = 0;

    // Loop over chunks of 4 doubles per AVX register
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vc = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&c[i], vc);
    }

    // remainder loop
    for (; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}


// SIMD: c[i] = a[i] * b[i]
void serial_simd_multiply(const std::vector<double>& a,
                   const std::vector<double>& b,
                   std::vector<double>& c)
{
    size_t n = a.size();
    size_t i = 0;

    // Loop over chunks of 4 doubles per AVX register
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vc = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&c[i], vc);
    }

    // remainder loop
    for (; i < n; i++)
    {
        c[i] = a[i] * b[i];
    }
}



// SIMD: c[i] = a[i] / b[i], with safe divide
void serial_simd_divide(const std::vector<double>& a,
                 const std::vector<double>& b,
                 std::vector<double>& c)
{
    size_t n = a.size();
    size_t i = 0;

    __m256d epsilon = _mm256_set1_pd(1e-9);

    // Loop over chunks of 4 doubles per AVX register
    for (; i + 4 <= n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        vb = _mm256_max_pd(vb, epsilon);      // avoid divide by zero
        __m256d vc = _mm256_div_pd(va, vb);
        _mm256_storeu_pd(&c[i], vc);
    }

    // remainder loop
    for (; i < n; i++)
    {
        c[i] = a[i] / std::max(b[i], 1e-9);
    }
}


void serial_simd_power(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    for(std::size_t i = 0; i <numbers_a.size(); i++)
    {
        numbers_c[i] = std::pow(numbers_a[i], numbers_b[i]);
    }
}



// Serial task - perform operation on numbers in the vector
void serial_simd_task(const std::string& operation, const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    if(operation == "add") serial_simd_add(numbers_a, numbers_b, numbers_c);
    if(operation == "multiply") serial_simd_multiply(numbers_a, numbers_b, numbers_c);
    if(operation == "divide") serial_simd_divide(numbers_a, numbers_b, numbers_c);
    if(operation == "power") serial_simd_power(numbers_a, numbers_b, numbers_c);
}





//======================================
// Serial
//======================================

void serial_add(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    for(std::size_t i = 0; i <numbers_a.size(); i++)
    {
        numbers_c[i] = numbers_a[i] + numbers_b[i];
    }
}


void serial_multiply(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    for(std::size_t i = 0; i <numbers_a.size(); i++)
    {
        numbers_c[i] = numbers_a[i] * numbers_b[i];
    }
}


void serial_divide(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    for(std::size_t i = 0; i <numbers_a.size(); i++)
    {
        numbers_c[i] = numbers_a[i] / std::max(numbers_b[i], 1.0e-9);
    }
}


void serial_power(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    for(std::size_t i = 0; i <numbers_a.size(); i++)
    {
        numbers_c[i] = std::pow(numbers_a[i], numbers_b[i]);
    }
}



// Serial task - perform operation on numbers in the vector
void serial_task(const std::string& operation, const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    if(operation == "add") serial_add(numbers_a, numbers_b, numbers_c);
    if(operation == "multiply") serial_multiply(numbers_a, numbers_b, numbers_c);
    if(operation == "divide") serial_divide(numbers_a, numbers_b, numbers_c);
    if(operation == "power") serial_power(numbers_a, numbers_b, numbers_c);
}






// Check the sum of an array (serial)
double check_sum(const std::vector<double>& numbers_c)
{
    double sum {};
    for(const auto& number : numbers_c)
    {
        sum = sum + number;
    }
    return sum;
}



int main(int argc, char** argv) 
{

    // Must have 4 arguments
    if (argc < 4) 
    {
        std::cerr << "Usage: " << argv[0] << " time_limit   vec_size   operation\n";
        return 1;
    }

    // Read in test_time and size of vector
    double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);
    const std::string operation = argv[3];


    if(!(operation == "add" || operation == "multiply" || operation == "divide" || operation == "power"))
    {
        std::cout << operation << std::endl;
        return 1;
    }

    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);    // [0.0, 1.0)

    // Vector of numbers
    std::vector<double> numbers_a;
    std::vector<double> numbers_b;
    numbers_a.reserve(N);
    numbers_b.reserve(N);

    // Populate vectors
    for (std::size_t i = 0; i < N; ++i) 
    {
        numbers_a.emplace_back(dist(rng));
        numbers_b.emplace_back(dist(rng));
    }

    double expected_value {};

    // Expected value
    {
        std::vector<double> numbers_c(N);
        serial_task(operation, numbers_a, numbers_b, numbers_c);
        expected_value = check_sum(numbers_c);
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
    std::vector<double> numbers_c(N);    
    do 
    {
        serial_simd_task(operation, numbers_a, numbers_b, numbers_c);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);

    // Clean up
    auto t2 = std::chrono::steady_clock::now();


    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    // Check
    double calculated_value = check_sum(numbers_c);
   
    double time_setup = std::chrono::duration<double>(t1 - t0).count();
    double time_calc = std::chrono::duration<double>(t2 - t1).count();
    double time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    double time_total = std::chrono::duration<double>(t3 - t0).count();
    double time_per_iteration = time_calc / iters;


    std::string method {"Serial"};
    std::string comments {"operation:" + operation};
    bool passed_check = std::abs(calculated_value - expected_value) < 1.0e-9;

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
