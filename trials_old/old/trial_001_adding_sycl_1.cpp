// sycl_sum.cpp
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>




// helper
static inline std::size_t round_up(std::size_t n, std::size_t m) {
    return ((n + m - 1) / m) * m;
}

long long task_fast(const std::vector<int>& numbers, sycl::queue& q)
{
    const std::size_t N = numbers.size();
    if (N == 0) return 0;

    constexpr std::size_t local  = 256;                 // tune per device
    const std::size_t     global = round_up(N, local);  // multiple of local

    // USM
    int*       d_in  = sycl::malloc_device<int>(N, q);
    long long* d_sum = sycl::malloc_shared<long long>(1, q);
    *d_sum = 0;

    // H2D
    q.memcpy(d_in, numbers.data(), N * sizeof(int)).wait();

    q.submit([&](sycl::handler& h) {
        // local scratch for block reduction
        sycl::local_accessor<long long, 1> lmem(local, h);

        h.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> it) {
                const std::size_t gid = it.get_global_linear_id();
                const std::size_t lid = it.get_local_linear_id();
                const std::size_t gsz = it.get_global_range(0);

                // strided accumulate to balance tails
                long long acc = 0;
                for (std::size_t i = gid; i < N; i += gsz) {
                    acc += static_cast<long long>(d_in[i]);
                }

                // write to local memory and reduce within work-group
                lmem[lid] = acc;
                it.barrier(sycl::access::fence_space::local_space);

                // tree reduction in local memory
                for (std::size_t stride = local / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        lmem[lid] += lmem[lid + stride];
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // one atomic add per group
                if (lid == 0) {
                    sycl::atomic_ref<
                        long long,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space
                    > atomic_sum(*d_sum);
                    atomic_sum.fetch_add(lmem[0]);
                }
            }
        );
    }).wait();

    long long result = *d_sum;
    sycl::free(d_in, q);
    sycl::free(d_sum, q);
    return result;
}

// If you want a drop-in wrapper like before:
long long task(const std::vector<int>& numbers) {
    static sycl::queue q{ sycl::default_selector_v,
                          sycl::property::queue::in_order{} };
    return task_fast(numbers, q);
}


int main(int argc, char** argv) 
{    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }
    
    const double test_time = std::atof(argv[1]);
    const int    N         = std::atoi(argv[2]);

    std::vector<int> numbers;
    numbers.reserve(N);

    int last = 0;
    for (int i = 0; i < N; ++i) {
        int n = 8039 * (last + i + 550607) % 10000;
        numbers.push_back(n);
        last = n;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    long long sum = 0;
    do {
        sum = task(numbers);
        ++iters;
    } while (std::chrono::steady_clock::now() < deadline);
      
    std::cout << "Sycl1," << iters << "," << sum << std::endl;
    return 0;
}
