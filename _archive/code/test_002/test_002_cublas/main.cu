#include <iostream>
#include <cmath>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>


int main() 
{


    auto start_setup = std::chrono::high_resolution_clock::now();

    const int N = 1024;
    const int size = N * N * sizeof(double);
    const size_t bytes = size * sizeof(double);

    // Allocate host memory
    double* h_A = new double[size];
    double* h_B = new double[size];
    double* h_C = new double[size];

    // Initialize A and B deterministically
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = (i + j) % 100;
            h_B[i * N + j] = (i * j + 3) % 100;
        }
    }

    auto end_setup = std::chrono::high_resolution_clock::now();


    auto start_calc = std::chrono::high_resolution_clock::now();
    //############################################################

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);


    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Constants for cublasDgemm
    const double alpha = 1.0;
    const double beta = 0.0;

    // Matrix multiplication: C = alpha * A * B + beta * C
    // cuBLAS uses column-major order, so arguments are transposed
    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,  // B comes first due to column-major order
                d_A, N,
                &beta,
                d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();



    // Sum result matrix
    double sum = 0.0;
    for (int i = 0; i < size; ++i)
    {
        sum += h_C[i];
    }

    // Calculate duration
    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << sum << std::endl;



    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
 
