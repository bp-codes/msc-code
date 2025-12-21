#include <iostream>
#include <cmath>
#include <chrono>

int main() 
{

    //############################################################
    //   Set up calculation
    //############################################################

    auto start_setup = std::chrono::high_resolution_clock::now();

    double timeout {10.0};
    constexpr int num_iterations = 1'000'000;
    double result {};

    auto end_setup = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration_setup = end_setup - start_setup;



    //############################################################
    //   Start calculation
    //############################################################

    auto start_calc = std::chrono::high_resolution_clock::now();
    size_t counter {};

    while(true)
    {
        result = 0.0;
        for (int i = 0; i < num_iterations; ++i) 
        {
            result += std::exp(static_cast<double>(i % 100) / 10.0); // avoid overflow
        }


        // Check time
        const auto now = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(now - start_calc).count();

        counter++;
        if (elapsed >= timeout) 
        {
            break;
        }
    }


    // Output results

    std::cout << duration_setup.count() << "," << counter << "," << result << std::endl;

    return 0;
}

