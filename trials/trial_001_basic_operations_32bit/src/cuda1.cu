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
__global__ void sum_kernel(const int* numbers, int* result, int N) 
{
    // shared data for block
    extern __shared__ int block_shared_data[];
    unsigned int thread_index = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;   // blockIdx.x block id, blockDim.x block size (256)

    // load input into shared memory (just load 0 if i >= N)
    block_shared_data[thread_index] = (i < N) ? numbers[i] : 0;

    __syncthreads();    //  Barrier

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)    // Halve s each loop
    {
        if (thread_index < s) 
        {
            block_shared_data[thread_index] += block_shared_data[thread_index + s];
        }
        __syncthreads();     //  Barrier
    }

    // write result for this block to global memory
    if (thread_index == 0) 
    {
        atomicAdd(result, block_shared_data[0]);
    }
}

// Wrapper for sum
int task(const int* device_numbers, int* device_result, int N) 
{
    // Set result on device equal to 0
    CUDA_CHECK(cudaMemset(device_result, 0, sizeof(int)));

    
    int block_size = 256;                                   // Number of threads in block (256 = 8 warps)
    int grid_size = (N + block_size - 1) / block_size;      // Total number of threads
    size_t block_shared_memory_size = block_size * sizeof(int);

    sum_kernel<<<grid_size, block_size, block_shared_memory_size>>>(device_numbers, device_result, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_sum = 0;
    CUDA_CHECK(cudaMemcpy(&h_sum, device_result, sizeof(int), cudaMemcpyDeviceToHost));
    return h_sum;
}


// Serial task - sum numbers in the vector
int serial_task(std::vector<int> numbers)
{
    int sum {};
    for(size_t i = 0; i <numbers.size(); i++)
    {
        sum = sum + numbers[i];
    }
    return sum;
}



int main(int argc, char** argv) 
{

    // Must have 3 arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }
    
    // Read in test_time and size of vector
    double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);

    // Vector of numbers
    std::vector<int> numbers;
    numbers.reserve(N);

    // Populate vector
    int last {};
    for (int i = 0; i < N; ++i) 
    {
        long long n = (i * 550607 + 8807 + last) % 109;
        numbers.push_back(n);
        last = n;
    }


    int expected_sum = serial_task(numbers);


    
    // ======= Calculation Starts ========

    auto t0 = std::chrono::steady_clock::now();

    // Allocate on device
    int* device_numbers = nullptr;
    int* device_result = nullptr;
    CUDA_CHECK(cudaMalloc(&device_numbers, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(  device_numbers, 
                            numbers.data(),
                            N * sizeof(int), 
                            cudaMemcpyHostToDevice));

    // Benchmark loop
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    int sum {};
    do {
        sum = task(device_numbers, device_result, N);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);
   
    // end time
    auto t2 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    CUDA_CHECK(cudaFree(device_numbers));
    CUDA_CHECK(cudaFree(device_result));

    // Actual end time (after cleanup)
    auto t3 = std::chrono::steady_clock::now();

   
    double time_taken = std::chrono::duration<double>(t2 - t1).count();
    double time_per_iteration = time_taken / iters;
    double setup_time = std::chrono::duration<double>(t1 - t0).count();
    double cleanup_time = std::chrono::duration<double>(t3 - t2).count();


    // Technology,ExpectedSum,Sum,Iterations,TimePerIteration,SetupTime,CleanupTime
    std::cout << "Cuda," << expected_sum << "," << sum << "," << iters << "," << time_per_iteration 
              << "," << setup_time <<  "," << cleanup_time << std::endl;

    return 0;
}
