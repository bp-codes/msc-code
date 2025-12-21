 #include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
extern "C" double exp(double);  

int main() {

    constexpr int num_iterations = 100'000'000;
    double result = 0.0;

    auto start_setup = std::chrono::high_resolution_clock::now();
    //############################################################

    // Allocate an array to store the results on host
    double* results = new double[num_iterations];

    //############################################################
    auto end_setup = std::chrono::high_resolution_clock::now();




    auto start_calc = std::chrono::high_resolution_clock::now();
    //############################################################

    // Offload to GPU: compute exponentials and store in array
    #pragma omp target teams distribute parallel for map(tofrom: results[0:num_iterations])
    for (int i = 0; i < num_iterations; ++i) {
        double x = static_cast<double>(i % 100) / 10.0;
        results[i] = exp(x);
    }

    // Accumulate result on host (can also be done on device if needed)
    for (int i = 0; i < num_iterations; ++i) {
        result += results[i];
    }

    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();
    
    
    // Calculate duration
    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << result << std::endl;

    return 0;
}

