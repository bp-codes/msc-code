// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
#include <sycl/sycl.hpp>



//======================================
// Sycl
//======================================



sycl::event sycl_add(std::size_t N, 
                sycl::queue& q,
                const double* sycldev_numbers_a, 
                const double* sycldev_numbers_b, 
                double* sycldev_numbers_c)
{
    return q.parallel_for(
        sycl::range<1>(N), [=](sycl::id<1> i) 
        {
            std::size_t idx = i[0];
            sycldev_numbers_c[idx] = sycldev_numbers_a[idx] + sycldev_numbers_b[idx];
        }
    );
}


sycl::event sycl_multiply(std::size_t N,
                sycl::queue& q,
                const double* sycldev_numbers_a,
                const double* sycldev_numbers_b,
                double* sycldev_numbers_c)
{
    return q.parallel_for(
        sycl::range<1>(N), [=](sycl::id<1> i) 
        {
            std::size_t idx = i[0];
            sycldev_numbers_c[idx] = sycldev_numbers_a[idx] * sycldev_numbers_b[idx];
        }
    );
}


sycl::event sycl_divide(std::size_t N,
                sycl::queue& q,
                const double* sycldev_numbers_a,
                const double* sycldev_numbers_b,
                double* sycldev_numbers_c)
{
    return q.parallel_for(
        sycl::range<1>(N), [=](sycl::id<1> i) 
        {
            std::size_t idx = i[0];
            sycldev_numbers_c[idx] = sycldev_numbers_a[idx] / sycldev_numbers_b[idx];
        }
    );
}


sycl::event sycl_power(std::size_t N, 
                sycl::queue& q,
                const double* sycldev_numbers_a, 
                const double* sycldev_numbers_b, 
                double* sycldev_numbers_c)
{
    return q.parallel_for(
        sycl::range<1>(N), [=](sycl::id<1> i) 
        {
            std::size_t idx = i[0];
            sycldev_numbers_c[idx] = sycl::pow(sycldev_numbers_a[idx], sycldev_numbers_b[idx]);
        }
    );
}


// Serial task - perform operation on numbers in the vector
sycl::event sycl_task( const std::string& operation, 
                std::size_t N, 
                sycl::queue& q,
                double* sycldev_numbers_a, 
                double* sycldev_numbers_b, 
                double* sycldev_numbers_c)
{
    if(operation == "add") return sycl_add(N, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
    if(operation == "multiply") sycl_multiply(N, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
    if(operation == "divide") sycl_divide(N, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
    if(operation == "power") sycl_power(N, q, sycldev_numbers_a, sycldev_numbers_b, sycldev_numbers_c);
}




//======================================
// Serial
//======================================


void serial_add(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    for(int i = 0; i <numbers_a.size(); i++)
    {
        numbers_c[i] = numbers_a[i] + numbers_b[i];
    }
}


void serial_multiply(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    for(int i = 0; i <numbers_a.size(); i++)
    {
        numbers_c[i] = numbers_a[i] * numbers_b[i];
    }
}


void serial_divide(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    for(int i = 0; i <numbers_a.size(); i++)
    {
        numbers_c[i] = numbers_a[i] / std::max(numbers_b[i], 1.0e-9);
    }
}


void serial_power(const std::vector<double>& numbers_a, const std::vector<double>& numbers_b, std::vector<double>& numbers_c)
{
    for(int i = 0; i <numbers_a.size(); i++)
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
    for (int i = 0; i < N; ++i) 
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

    
    sycl::queue q{ sycl::default_selector_v };
    std::cerr << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Results
    std::vector<double> numbers_c(N);    

    // Allocate device memory once
    double* sycldev_numbers_a = sycl::malloc_device<double>(N, q);
    double* sycldev_numbers_b = sycl::malloc_device<double>(N, q);
    double* sycldev_numbers_c = sycl::malloc_device<double>(N, q);

    // Copy once
    q.memcpy(sycldev_numbers_a, numbers_a.data(), N * sizeof(double)).wait();
    q.memcpy(sycldev_numbers_b, numbers_b.data(), N * sizeof(double)).wait();



    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    int sum {};
    sycl::event last;

    // Do as many times as possible before time runs out
    do 
    {
        last = sycl_task( operation,   
                   N,    
                   q,   
                   sycldev_numbers_a, 
                   sycldev_numbers_b, 
                   sycldev_numbers_c );
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);

    last.wait();

    // Clean up
    auto t2 = std::chrono::steady_clock::now();

    // Copy results
    q.memcpy(numbers_c.data(), sycldev_numbers_c, N * sizeof(double)).wait();

    // Free device allocations
    sycl::free(sycldev_numbers_a, q);
    sycl::free(sycldev_numbers_b, q);
    sycl::free(sycldev_numbers_c, q);

    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    double calculated_value = check_sum(numbers_c);

   
    double time_setup = std::chrono::duration<double>(t1 - t0).count();
    double time_calc = std::chrono::duration<double>(t2 - t1).count();
    double time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    double time_total = std::chrono::duration<double>(t3 - t0).count();
    double time_per_iteration = time_calc / iters;


    std::string method {"SYCL"};
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
