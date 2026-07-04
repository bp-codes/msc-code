/*
g++-13 -O3 -std=c++20 -fopenmp \
    -fno-stack-protector \
    -fcf-protection=none \
    -foffload=nvptx-none \
    -foffload-options=nvptx-none=-misa=sm_80 \
    vector_add.cpp -o vector_add.x
    
OMP_TARGET_OFFLOAD=MANDATORY ./vector_add.x
*/


#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

int main()
{
    const std::size_t n = 1'000'000;

    std::vector<double> a(n);
    std::vector<double> b(n);
    std::vector<double> c(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<double>(i);
        b[i] = 2.0 * static_cast<double>(i);
    }

    double* a_ptr = a.data();
    double* b_ptr = b.data();
    double* c_ptr = c.data();

    #pragma omp target teams distribute parallel for \
        map(to: a_ptr[0:n], b_ptr[0:n]) \
        map(from: c_ptr[0:n])
    for (std::size_t i = 0; i < n; ++i) {
        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) {
        const double expected = 3.0 * static_cast<double>(i);
        if (std::abs(c[i] - expected) > 1.0e-12) {
            ok = false;
            std::cerr << "Mismatch at " << i
                      << ": got " << c[i]
                      << ", expected " << expected << '\n';
            break;
        }
    }

    std::cout << "OpenMP target devices: "
              << omp_get_num_devices() << '\n';

    std::cout << (ok ? "Success\n" : "Failed\n");

    return ok ? 0 : 1;
}