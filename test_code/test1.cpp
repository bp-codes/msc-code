// g++-13 -O3 -std=c++20 -fopenmp -fcf-protection=none test.cpp -o omp_gpu

#include <iostream>
#include <omp.h>

int main()
{
    int is_device = -1;

    #pragma omp target map(from:is_device)
    {
        is_device = omp_is_initial_device();
    }

    if (is_device == 0) {
        std::cout << "Running on GPU/offload device\n";
    } else {
        std::cout << "Running on host CPU\n";
    }

    std::cout << "Number of target devices: "
              << omp_get_num_devices() << "\n";
}