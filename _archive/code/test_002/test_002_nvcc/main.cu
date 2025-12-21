#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

__global__ void matmul(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

int main() 
{


    auto start_setup = std::chrono::high_resolution_clock::now();

    const int N = 1024;
    const int size = N * N * sizeof(double);

    // Allocate host memory
    double* h_A = new double[N * N];
    double* h_B = new double[N * N];
    double* h_C = new double[N * N];

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
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy A and B to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);
    matmul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();



    // Sum result matrix
    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
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

    return 0;
}
 
