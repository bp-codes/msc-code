// sycl.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
  #include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
  #include <CL/sycl.hpp>
#else
  #error "SYCL header not found. Use a SYCL 2020 compiler (oneAPI/dpcpp or hipSYCL)."
#endif

static void init(std::vector<float>& x,
                 std::vector<float>& y,
                 std::vector<float>& z) {
    const std::size_t N = x.size();
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = std::sin(0.001f * static_cast<float>(i)) + 1.0f;
        y[i] = std::cos(0.001f * static_cast<float>(i)) + 2.0f;
        z[i] = std::sin(0.002f * static_cast<float>(i)) + 3.0f;
    }
}

int main(int argc, char** argv) {
    std::size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ull << 24);
    double seconds_target = (argc > 2) ? std::atof(argv[2]) : 2.0;
    int inner_iters = (argc > 3) ? std::atoi(argv[3]) : 8;

    const float a = 1.0001f, b = 1.0002f, c = 0.9999f, d = 0.1234f;

    std::vector<float> hx(N), hy(N), hz(N);
    init(hx, hy, hz);

    sycl::queue q;
    try {
        q = sycl::queue{sycl::gpu_selector_v};
        std::cout << "SYCL benchmark on device: "
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";
    } catch (...) {
        std::cerr << "GPU not found; using default device.\n";
        q = sycl::queue{sycl::default_selector_v};
    }

    float* dx = sycl::malloc_device<float>(N, q);
    float* dy = sycl::malloc_device<float>(N, q);
    float* dz = sycl::malloc_device<float>(N, q);
    q.memcpy(dx, hx.data(), N * sizeof(float));
    q.memcpy(dy, hy.data(), N * sizeof(float));
    q.memcpy(dz, hz.data(), N * sizeof(float));
    q.wait();

    const std::size_t L = 256;
    const std::size_t G = ((N + L - 1) / L) * L;

    auto kernel_once = [&](int innerIters) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(G),
                                             sycl::range<1>(L)),
                           [=](sycl::nd_item<1> it) {
                std::size_t i = it.get_global_linear_id();
                if (i < N) {
                    float yi = dy[i];
                    const float xi = dx[i];
                    const float zi = dz[i];
                    for (int k = 0; k < innerIters; ++k) {
                        yi = a * yi + b * xi + c * zi + d;
                    }
                    dy[i] = yi;
                }
            });
        }).wait();
    };

    kernel_once(inner_iters);

    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(seconds_target);
    std::uint64_t iters = 0;
    do {
        kernel_once(inner_iters);
        ++iters;
    } while (std::chrono::steady_clock::now() < deadline);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    q.memcpy(hy.data(), dy, N * sizeof(float)).wait();
    double checksum = 0.0;
    for (std::size_t i = 0; i < N; ++i) checksum += hy[i];

    const double flops_per_elem = 7.0 * inner_iters;
    const double total_flops = static_cast<double>(iters) * static_cast<double>(N) * flops_per_elem;
    const double gflops = total_flops / 1e9 / secs;

    const double bytes_per_iter = static_cast<double>(N) * 16.0;
    const double gbytes_per_s = (bytes_per_iter * static_cast<double>(iters)) / 1e9 / secs;

    std::cout << std::fixed << std::setprecision(3)
              << "Time: " << secs << " s, Iters: " << iters
              << ", GFLOP/s: " << gflops
              << ", GB/s (effective): " << gbytes_per_s
              << ", checksum: " << checksum << "\n";

    sycl::free(dx, q);
    sycl::free(dy, q);
    sycl::free(dz, q);
    return 0;
}
 
