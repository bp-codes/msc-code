#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

__global__ void compute_exponentials(double* results, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double x = static_cast<double>(i % 100) / 10.0;
        results[i] = exp(x); // device math function
    }
}

int main() 
{

    //############################################################
    //   Set up calculation
    //############################################################

    auto start_setup = std::chrono::high_resolution_clock::now();

    double timeout {10.0};
    constexpr int num_iterations = 1'000'000;
    double result {};
    size_t size = num_iterations * sizeof(double);

    // Allocate memory
    double* d_results;
    double* h_results = new double[num_iterations];
    cudaMalloc(&d_results, size);

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto end_setup = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> duration_setup = end_setup - start_setup;




    //############################################################
    //   Start calculation
    //############################################################

    auto start_calc = std::chrono::high_resolution_clock::now();
    size_t counter {};

    while(true)
    {

        // Launch kernel
        int blockSize = 256;
        int gridSize = (num_iterations + blockSize - 1) / blockSize;

        cudaEventRecord(start);
        compute_exponentials<<<gridSize, blockSize>>>(d_results, num_iterations);
        cudaEventRecord(stop);

        // Wait for GPU
        cudaDeviceSynchronize();

        // Copy back results
        cudaMemcpy(h_results, d_results, size, cudaMemcpyDeviceToHost);

        // Compute dummy result to avoid optimization
        double result = 0.0;
        for (int i = 0; i < num_iterations; ++i)
        {
            result += h_results[i];
        }

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

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



    // Cleanup
    delete[] h_results;
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
 
