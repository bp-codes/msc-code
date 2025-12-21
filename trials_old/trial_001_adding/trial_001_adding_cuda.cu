// cuda.cu
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA error checking helper
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

// Kernel: parallel reduction to compute sum
__global__ void sumKernel(const int* numbers, int* result, int N) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load input into shared memory
    sdata[tid] = (i < N) ? numbers[i] : 0;
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// Wrapper for sum
int task(const int* d_numbers, int* d_result, int N) {
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(int)));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t shmSize = blockSize * sizeof(int);

    sumKernel<<<gridSize, blockSize, shmSize>>>(d_numbers, d_result, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_sum = 0;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    return h_sum;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }
    
    double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);

    // Generate numbers on host
    std::vector<int> numbers;
    numbers.reserve(N);

    int last = 0;
    for (int i = 0; i < N; ++i) {
        int n = 8039 * (last + i + 550607) % 10000;
        numbers.push_back(n);
        last = n;
    }

    // Allocate on device
    int* d_numbers = nullptr;
    int* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_numbers, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_numbers, numbers.data(),
                          N * sizeof(int), cudaMemcpyHostToDevice));

    // Benchmark loop
    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    int sum {};
    do {
        sum = task(d_numbers, d_result, N);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);
   
    std::cout << "Cuda," << iters << "," << sum << std::endl;

    CUDA_CHECK(cudaFree(d_numbers));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}
