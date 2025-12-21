// sycl_stopping_power_usm.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>

#include <sycl/sycl.hpp>

// Use std::log on host, sycl::log on device (so the same function works both places)
static inline float my_log(float x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::log(x);
#else
    return std::log(x);
#endif
}

static inline float my_sqrt(float x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::sqrt(x);
#else
    return std::sqrt(x);
#endif
}


static inline float stopping_power_fe(const float v_m_per_s) 
{
    static constexpr float c     = 299792458.0f;
    static constexpr float me    = 0.51099895000f;
    static constexpr float K     = 0.307075f;

    static constexpr float Z     = 26.0f;
    static constexpr float A     = 55.845f;
    static constexpr float rho   = 7.874f;
    static constexpr float I     = 286.0e-6f;

    static constexpr float M     = 938.2720813f;

    const float beta  = v_m_per_s / c;
    const float beta2 = beta * beta;
    const float inv_1_minus_beta2 = 1.0f / (1.0f - beta2);
    const float gamma2 = inv_1_minus_beta2;              // gamma^2
    const float gamma  = my_sqrt(gamma2);             // or std::sqrt on host via my_* wrappers

    const float me_over_M = me / M;

    const float numerator = 2.0f * me * beta2 * gamma2;
    const float denom = 1.0f + 2.0f * gamma * me_over_M + (me_over_M * me_over_M);
    const float Tmax = numerator / denom;

    const float logArg = (2.0f * me * beta2 * gamma2 * Tmax) / (I * I);

    const float prefactor = K * (Z / A) / beta2;
    const float bracket   = 0.5f * my_log(logArg) - beta2;

    return (prefactor * bracket) * rho;
}



// Serial: compute sum directly (matches host sum of per-element values)
float serial_expected_sum(const float* velocity_array, std::size_t n) 
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += stopping_power_fe(velocity_array[i]);
    }
    return sum;
}

// Serial: fill per-particle stopping power array (optional helper)
void serial_task(std::size_t n, const float* velocity_array, float* stopping_power) {
    for (std::size_t i = 0; i < n; ++i) 
    {
        stopping_power[i] = stopping_power_fe(velocity_array[i]);
    }
}

// SYCL: fill per-particle stopping power array on device
sycl::event sycl_task(
    sycl::queue& q,
    std::size_t n,
    const float* velocity_array,
    float* stopping_power
) {
  return q.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
      const std::size_t i = it.get_linear_id();
      stopping_power[i] = stopping_power_fe(velocity_array[i]);
  });
}

int main(int argc, char** argv) {
    if (argc < 3) 
    {
        std::cerr << "Usage: " << argv[0] << " time_limit   vec_size\n";
        return 1;
    }

    const double test_time = std::atof(argv[1]);
    const std::size_t N = static_cast<std::size_t>(std::atoll(argv[2]));

    // Setup timing start
    auto t0 = std::chrono::steady_clock::now();

    sycl::queue q{sycl::default_selector_v};
    std::cerr << "Using device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Allocate 
    float* velocity_host = sycl::malloc_shared<float>(N, q);
    float* stopping_power_host = sycl::malloc_shared<float>(N, q);

    float* velocity_device = sycl::malloc_device<float>(N, q);
    float* stopping_power_device = sycl::malloc_device<float>(N, q);

    if (!velocity_device || !stopping_power_device || !velocity_host || !stopping_power_host) 
    {
        std::cerr << "Memory allocation failed\n";
        return 2;
    }

    // Fill input once (host writes shared memory)
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < N; ++i) 
    {
        velocity_host[i] = 1.0e6f * dist(rng);
    }

    // Expected serial value (host)
    const float expected_value = serial_expected_sum(velocity_host, N);
    std::cout << "Serial computed expected value:  " << expected_value << "\n";

    q.memcpy(velocity_device, velocity_host, sizeof(float) * N).wait();

    // Calc timing start
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;


    // Run as many iterations as possible
    sycl::event last;
    do 
    {
        last = sycl_task(q, N, velocity_device, stopping_power_device);
        ++iters;
    } while (std::chrono::steady_clock::now() < deadline);

    // Ensure last submitted kernel finished before reading result
    last.wait();

    auto t2 = std::chrono::steady_clock::now();


    q.memcpy(stopping_power_host,
         stopping_power_device,
         sizeof(float) * N).wait();

    // Sum stopping_power on host for comparison
    float calculated_value = 0.0f;
    for (std::size_t i = 0; i < N; ++i) 
    {
        calculated_value += stopping_power_host[i];
    }

    // Free USM
    sycl::free(stopping_power_device, q);
    sycl::free(stopping_power_host, q);
    sycl::free(velocity_device, q);
    sycl::free(velocity_host, q);



    auto t3 = std::chrono::steady_clock::now();

    const double time_setup   = std::chrono::duration<double>(t1 - t0).count();
    const double time_calc    = std::chrono::duration<double>(t2 - t1).count();
    const double time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    const double time_total   = std::chrono::duration<double>(t3 - t0).count();
    const double time_per_iteration =
        (iters > 0) ? (time_calc / static_cast<double>(iters)) : 0.0;

    const bool passed_check =
        std::abs(calculated_value - expected_value) < 1.0e-6f;

    std::string method{"SYCL_USM_array"};
    std::string comments{"stopping_power"};

    std::cout << method << ","
              << expected_value << ","
              << calculated_value << ","
              << iters << ","
              << time_per_iteration << ","
              << time_setup << ","
              << time_calc << ","
              << time_cleanup << ","
              << time_total << ","
              << passed_check << ","
              << comments
              << "\n";

    return 0;
}
