// gpu_reuse.cpp
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

// helper: round n up to multiple of m
static inline std::size_t round_up(std::size_t n, std::size_t m) {
    return ((n + m - 1) / m) * m;
}

long long task_fast(int* d_in, std::size_t N, sycl::queue& q) {
    constexpr std::size_t local  = 256;
    const std::size_t     global = round_up(N, local);

    // shared (host-visible) scalar for the reduction result
    long long* d_sum = sycl::malloc_shared<long long>(1, q);

    q.submit([&](sycl::handler& h) {
        auto red = sycl::reduction(
            d_sum,
            sycl::plus<long long>(),
            sycl::property::reduction::initialize_to_identity{}
        );

        h.parallel_for(
            sycl::nd_range<1>(global, local),
            red,
            [=](sycl::nd_item<1> it, auto& sum) {
                const std::size_t i = it.get_global_linear_id();
                if (i < N) sum.combine(static_cast<long long>(d_in[i]));
            }
        );
    }).wait();

    
    long long result = *d_sum;  // read value while still allocated
    sycl::free(d_sum, q);
    return result;
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

    int last{};
    for (int i = 0; i < N; ++i) {
        int n = 8039 * (last + i + 550607) % 10000;
        numbers.push_back(n);
        last = n;
    }

    sycl::queue q{ sycl::default_selector_v };

    // Allocate device memory once
    int* d_in = sycl::malloc_device<int>(N, q);

    // Copy once
    q.memcpy(d_in, numbers.data(), N * sizeof(int)).wait();

    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    // Reuse d_in on every iteration
    int sum {};
    do {
        sum = task_fast(d_in, N, q);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);
   
    std::cout << "Sycl2," << iters << "," << sum << std::endl;

    sycl::free(d_in, q); // free at end

    return 0;
}
