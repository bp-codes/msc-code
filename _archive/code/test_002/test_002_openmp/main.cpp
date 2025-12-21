#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <omp.h> 

int main() 
{


    auto start_setup = std::chrono::high_resolution_clock::now();

    const int N = 1024;

    // Allocate matrices A, B, and C
    std::vector<std::vector<double>> A(N, std::vector<double>(N));
    std::vector<std::vector<double>> B(N, std::vector<double>(N));
    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j) 
        {
            A[i][j] = (i + j) % 100;
            B[i][j] = (i * j + 3) % 100;
        }
    }

    auto end_setup = std::chrono::high_resolution_clock::now();


    auto start_calc = std::chrono::high_resolution_clock::now();
    //############################################################

    // Matrix multiplication: C = A * B
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];

    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();

    // Sum all elements of C
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            sum += C[i][j];
    
    // Calculate duration
    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << sum << std::endl;

    return 0;
}

