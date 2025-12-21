#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <omp.h>

int main() 
{
    const int N = 1024;
    auto start_setup = std::chrono::high_resolution_clock::now();

    // Allocate and initialize matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (i + j) % 100;
            B[i * N + j] = (i * j + 3) % 100;
        }

    // Get raw pointers for OpenMP map clause
    double* a_ptr = A.data();
    double* b_ptr = B.data();
    double* c_ptr = C.data();

    auto end_setup = std::chrono::high_resolution_clock::now();
    auto start_calc = std::chrono::high_resolution_clock::now();

    // Offload matrix multiplication to GPU
    #pragma omp target data map(to: a_ptr[0:N*N], b_ptr[0:N*N]) map(from: c_ptr[0:N*N])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k)
                    sum += a_ptr[i * N + k] * b_ptr[k * N + j];
                c_ptr[i * N + j] = sum;
            }
    }

    auto end_calc = std::chrono::high_resolution_clock::now();

    // Sum result on host
    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
        sum += C[i];

    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," 
              << duration_calc.count() << "," 
              << sum << std::endl;

    return 0;
}
