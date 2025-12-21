// openmp.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <omp.h>

static void init(std::vector<float>& x,
                 std::vector<float>& y,
                 std::vector<float>& z) {
    const std::size_t N = x.size();
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        x[i] = std::sin(0.001f * static_cast<float>(i)) + 1.0f;
        y[i] = std::cos(0.001f * static_cast<float>(i)) + 2.0f;
        z[i] = std::sin(0.002f * static_cast<float>(i)) + 3.0f;
    }
}

static void compute_once(float* __restrict y,
                         const float* __restrict x,
                         const float* __restrict z,
                         std::size_t N,
                         float a, float b, float c, float d,
                         int inner_iters) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        float yi = y[i];
        const float xi = x[i];
        const float zi = z[i];
        #pragma omp simd
        for (int k = 0; k < inner_iters; ++k) {
            yi = a * yi + b * xi + c * zi + d;
        }
        y[i] = yi;
    }
}

int main(int argc, char** argv) {
    std::size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ull << 24);
    double seconds_target = (argc > 2) ? std::atof(argv[2]) : 2.0;
    int inner_iters = (argc > 3) ? std::atoi(argv[3]) : 8;

    const float a = 1.0001f, b = 1.0002f, c = 0.9999f, d = 0.1234f;

    std::vector<float> x(N), y(N), z(N);
    init(x, y, z);

    std::cout << "OpenMP benchmark (threads=" << omp_get_max_threads() << ")\n"
              << "N=" << N << ", seconds_target=" << seconds_target
              << ", inner_iters=" << inner_iters << "\n";

    compute_once(y.data(), x.data(), z.data(), N, a, b, c, d, inner_iters);

    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(seconds_target);
    std::uint64_t iters = 0;
    do {
        compute_once(y.data(), x.data(), z.data(), N, a, b, c, d, inner_iters);
        ++iters;
    } while (std::chrono::steady_clock::now() < deadline);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    double checksum = 0.0;
    #pragma omp parallel for reduction(+:checksum) schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) checksum += y[i];

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
    return 0;
}
 
