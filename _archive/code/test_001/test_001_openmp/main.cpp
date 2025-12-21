#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

int main() 
{


    auto start_setup = std::chrono::high_resolution_clock::now();

    constexpr int num_iterations = 100'000'000;
    double result = 0.0;

    auto end_setup = std::chrono::high_resolution_clock::now();


    auto start_calc = std::chrono::high_resolution_clock::now();
    //############################################################


    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < num_iterations; ++i) 
    {
        result += std::exp(static_cast<double>(i % 100) / 10.0); // avoid overflow
    }



    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();
    
    // Calculate duration
    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << result << std::endl;

    return 0;
}


