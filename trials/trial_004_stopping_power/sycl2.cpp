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
static inline double my_log(double x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::log(x);
#else
    return std::log(x);
#endif
}

double stopping_power_fe(double v_m_per_s) {
    // Physical constants
    const double c = 299792458.0;                 // m/s
    const double me_MeV = 0.51099895000;          // electron mass energy, MeV
    const double K = 0.307075;                    // MeV·cm^2/g (Bethe constant)

    // Iron material constants
    const double Z = 26.0;                        // atomic number
    const double A = 55.845;                      // g/mol
    const double rho = 7.874;                     // g/cm^3
    const double I_eV = 286.0;                    // mean excitation energy, eV
    const double I_MeV = I_eV * 1.0e-6;           // convert eV -> MeV

    // Projectile (proton)
    const double z = 1.0;                         // charge state in |e|
    const double M_MeV = 938.2720813;             // proton rest energy, MeV

    // Guard: valid speed
    if (!(v_m_per_s > 0.0) || !(v_m_per_s < c)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Relativistic kinematics
    const double beta  = v_m_per_s / c;
    const double beta2 = beta * beta;
    const double gamma = 1.0 / std::sqrt(1.0 - beta2);
    const double me_over_M = me_MeV / M_MeV;

    // Tmax for heavy particle-electron kinematics (MeV)
    const double numerator = 2.0 * me_MeV * beta2 * gamma * gamma;
    const double denom = 1.0 + 2.0 * gamma * me_over_M + (me_over_M * me_over_M);
    const double Tmax = numerator / denom;

    // log argument
    const double logArg =
        (2.0 * me_MeV * beta2 * gamma * gamma * Tmax) / (I_MeV * I_MeV);
    if (!(logArg > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Mass stopping power (MeV·cm^2/g)
    const double prefactor = K * (Z / A) * (z * z) / beta2;
    const double bracket   = 0.5 * my_log(logArg) - beta2;
    const double S_over_rho = prefactor * bracket;

    // Linear stopping power in MeV/cm
    const double S_linear = S_over_rho * rho;
    return S_linear;
}

// Serial: compute sum directly (matches host sum of per-element values)
double serial_expected_sum(const double* velocity_array, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += stopping_power_fe(velocity_array[i]);
    }
    return sum;
}

// Serial: fill per-particle stopping power array (optional helper)
void serial_task(std::size_t n, const double* velocity_array, double* stopping_power) {
    for (std::size_t i = 0; i < n; ++i) 
    {
        stopping_power[i] = stopping_power_fe(velocity_array[i]);
    }
}

// SYCL: fill per-particle stopping power array on device
sycl::event sycl_task(
    sycl::queue& q,
    std::size_t n,
    const double* velocity_array,
    double* stopping_power
) {
    return q.parallel_for(
        sycl::range<1>(n),
        [=](sycl::id<1> idx) {
            const std::size_t i = idx[0];
            stopping_power[i] = stopping_power_fe(velocity_array[i]);
        }
    );
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

    // Allocate USM shared
    double* velocity_array = sycl::malloc_shared<double>(N, q);
    double* stopping_power = sycl::malloc_shared<double>(N, q);

    if (!velocity_array || !stopping_power) 
    {
        std::cerr << "USM allocation failed\n";
        return 2;
    }

    // Fill input once (host writes shared memory)
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (std::size_t i = 0; i < N; ++i) 
    {
        velocity_array[i] = 1.0e6 * dist(rng);
    }

    // Expected serial value (host)
    const double expected_value = serial_expected_sum(velocity_array, N);
    std::cout << "Serial computed expected value:  " << expected_value << "\n";

    // Calc timing start
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    // Run as many iterations as possible
    sycl::event last;
    do 
    {
        last = sycl_task(q, N, velocity_array, stopping_power);
        ++iters;
    } while (std::chrono::steady_clock::now() < deadline);

    // Ensure last submitted kernel finished before reading result
    last.wait();

    auto t2 = std::chrono::steady_clock::now();

    // Sum stopping_power on host for comparison
    double calculated_value = 0.0;
    for (std::size_t i = 0; i < N; ++i) 
    {
        calculated_value += stopping_power[i];
    }

    auto t3 = std::chrono::steady_clock::now();

    // Free USM
    sycl::free(stopping_power, q);
    sycl::free(velocity_array, q);

    const double time_setup   = std::chrono::duration<double>(t1 - t0).count();
    const double time_calc    = std::chrono::duration<double>(t2 - t1).count();
    const double time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    const double time_total   = std::chrono::duration<double>(t3 - t0).count();
    const double time_per_iteration =
        (iters > 0) ? (time_calc / static_cast<double>(iters)) : 0.0;

    const bool passed_check =
        std::abs(calculated_value - expected_value) < 1.0e-6;

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
