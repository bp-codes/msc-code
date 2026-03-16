// cuda.cu
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

/* -------------------------------------------------------------
   Kernel – works for any size that fits in ONE block (≤ block_size)
   ------------------------------------------------------------- */
__global__ void reduce_one_block(const double* __restrict__ in,
                                 double* __restrict__ out,
                                 int N)
{
    extern __shared__ double shared_data[];

    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x + tid;

    // 1. Load – pad with 0.0 for out-of-range threads
    shared_data[tid] = (idx < N) ? in[idx] : 0.0;
    __syncthreads();

    // 2. Warp-level reduction
    double val = shared_data[tid];
    for (int offset = 16; offset > 0; offset >>= 1)   // Halves offset with each loop
    {
        val += __shfl_down_sync(0xffffffff, val, offset);      // 0xffffffff all 32 threads, thread value: val, how far to read
    }

    // Write warp sum to shared data
    shared_data[tid] = val;
    __syncthreads();

    // 3. Reduce the 8 warp sums (256/32 = 8) using the first warp
    //    8 warps → 8 values at shared_data[0,32,64,…]
    if (tid < 8) 
    {                     
        val = shared_data[tid * 32];
        for (int offset = 4; offset > 0; offset >>= 1) 
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) 
        {
            out[blockIdx.x] = val;     // each block writes its own sum
        }
    }
}

// Host wrapper – replace your cuda_task with this
double cuda_task(const double* d_input, int N)
{
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    double* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));

    if (num_blocks == 1) {
        reduce_one_block<<<1, block_size, block_size * sizeof(double)>>>(
            d_input, d_result, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        double host_result = 0.0;
        CUDA_CHECK(cudaMemcpy(&host_result, d_result, sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_result));
        return host_result;
    }

    // --- Allocate and ZERO block sums ---
    double* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_block_sums, 0, num_blocks * sizeof(double)));

    // --- First pass: reduce input to block sums ---
    reduce_one_block<<<num_blocks, block_size, block_size * sizeof(double)>>>(
        d_input, d_block_sums, N);
    CUDA_CHECK(cudaGetLastError());

    // --- Second pass: reduce block sums ---
    if (num_blocks <= block_size) {
        // Fits in shared memory
        reduce_one_block<<<1, block_size, block_size * sizeof(double)>>>(
            d_block_sums, d_result, num_blocks);
    } else {
        // Use global memory reduction (simple tree)
        int remaining = num_blocks;
        double* d_src = d_block_sums;
        double* d_dst = d_result;

        while (remaining > 1) {
            int blocks = (remaining + block_size - 1) / block_size;
            double* d_temp = nullptr;
            if (blocks > 1) {
                CUDA_CHECK(cudaMalloc(&d_temp, blocks * sizeof(double)));
                CUDA_CHECK(cudaMemset(d_temp, 0, blocks * sizeof(double)));
            } else {
                d_temp = d_result;
            }

            reduce_one_block<<<blocks, block_size, block_size * sizeof(double)>>>(
                d_src, d_temp, remaining);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            if (d_src != d_block_sums && d_src != d_result) {
                CUDA_CHECK(cudaFree(d_src));
            }
            d_src = d_temp;
            remaining = blocks;
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double host_result = 0.0;
    CUDA_CHECK(cudaMemcpy(&host_result, d_result, sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_block_sums));

    return host_result;
}


// Serial task - sum numbers in the vector
double serial_task(const std::vector<double>& numbers)
{
    auto sum {0.0};
    for(const auto val : numbers)
    {
        sum += val;
    }
    return sum;
}



int main(int argc, char** argv) 
{

    // Must have 3 arguments
    if (argc < 3) 
    {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }
    
    // Read in test_time and size of vector
    const double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);
    const std::string operation = "Sum vector elements.";

    if(N <= 0)
    {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }
    
    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);  // [0.0, 1.0)

    // Vector of numbers
    std::vector<double> numbers;
    numbers.reserve(N);

    // Populate vector
    for (int i = 0; i < N; ++i) 
    {
        numbers.emplace_back(dist(rng));
    }

    auto expected_value = serial_task(numbers);

    
    // ======= Calculation Starts ========

    auto t0 = std::chrono::steady_clock::now();

    // Allocate on device
    double* device_numbers = nullptr;
    CUDA_CHECK(cudaMalloc(&device_numbers, N * sizeof(double)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(  device_numbers, 
                            numbers.data(),
                            N * sizeof(double), 
                            cudaMemcpyHostToDevice));

    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    double calculated_value {};

    do 
    {
        calculated_value = cuda_task(device_numbers, N);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);
   
    // Clean up
    auto t2 = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaFree(device_numbers));

    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========
   
    auto time_setup = std::chrono::duration<double>(t1 - t0).count();
    auto time_calc = std::chrono::duration<double>(t2 - t1).count();
    auto time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    auto time_total = std::chrono::duration<double>(t3 - t0).count();
    auto time_per_iteration = time_calc / iters;

    std::string method {"CUDA"};
    std::string device {"gpu"};
    std::string comments {"operation:" + operation};
    bool passed_check = std::abs(calculated_value - expected_value) < 1.0e-9;

   
    // Technology,Iterations,Sum,TimePerIteration,SetupTime
    std::cout << method << "," 
              << device << ","
              << std::scientific << std::setprecision(12)
              << expected_value << "," 
              << calculated_value << "," 
              << std::scientific << std::setprecision(6)
              << iters << "," 
              << time_per_iteration << "," 
              << time_setup << "," 
              << time_calc << "," 
              << time_cleanup << "," 
              << time_total << "," 
              << passed_check << "," 
              << comments << "," 
              << "" << "" 
              << std::endl;

    return 0;

}
