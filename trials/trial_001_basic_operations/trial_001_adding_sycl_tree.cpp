// gpu_reuse.cpp
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>


// round up to a multiple of 'm'
static inline std::size_t round_up(std::size_t n, std::size_t m) {
    return (n + m - 1) / m * m;
}


long long task_fast(int* d_in, std::size_t N, sycl::queue& q) {
    constexpr std::size_t local  = 256;
    const std::size_t     global = round_up(N, local);

    // host-visible result
    long long* d_sum = sycl::malloc_shared<long long>(1, q);
    *d_sum = 0; // important: we use atomic adds

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<long long, 1> scratch(local, h);

        h.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> it) {
                const std::size_t gid = it.get_global_linear_id();
                const std::size_t lid = it.get_local_linear_id();

                // load to local memory (guard OOB)
                long long x = (gid < N) ? static_cast<long long>(d_in[gid]) : 0LL;
                scratch[lid] = x;
                it.barrier(sycl::access::fence_space::local_space);

                // tree reduction in local memory
                for (std::size_t stride = local >> 1; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        scratch[lid] += scratch[lid + stride];
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // one atomic add per work-group
                if (lid == 0) {
                    sycl::atomic_ref<
                        long long,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space
                    > a(*d_sum);
                    a.fetch_add(scratch[0]);
                }
            }
        );
    }).wait();

    long long result = *d_sum;
    sycl::free(d_sum, q);
    return result;
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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }
    
    double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);

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

    sycl::queue q{ sycl::default_selector_v };

    // Allocate device memory once
    int* d_in = sycl::malloc_device<int>(N, q);

    // Copy once
    q.memcpy(d_in, numbers.data(), N * sizeof(int)).wait();

    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    // Reuse d_in on every iteration
    int sum {};
    do {
        sum = task_fast(d_in, N, q);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);

    // end time
    auto t2 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    sycl::free(d_in, q); // free at end

    // Actual end time (after cleanup)
    auto t3 = std::chrono::steady_clock::now();

   
    double time_taken = std::chrono::duration<double>(t2 - t1).count();
    double time_per_iteration = time_taken / iters;
    double setup_time = std::chrono::duration<double>(t1 - t0).count();
    double cleanup_time = std::chrono::duration<double>(t3 - t2).count();


    // Technology,Iterations,Sum,TimePerIteration,SetupTime,CleanupTime
    std::cout << "Cuda," << expected_sum << "," << sum << "," << iters << "," << time_per_iteration 
              << "," << setup_time <<  "," << cleanup_time << std::endl;

    return 0;
}
