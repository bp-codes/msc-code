// cuda

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/Error.hpp"
#include "helper/helper_cuda.hpp"
#include "nlohmann/json.hpp"

// -----------------------------
// CUDA error checking
// -----------------------------
#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                        \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " ("               \
                      << static_cast<int>(err__) << ") at " << __FILE__ << ":" << __LINE__ \
                      << "\n";                                                             \
            std::exit(2);                                                                  \
        }                                                                                  \
    } while (0)

// -----------------------------
// Host/device math helpers
// -----------------------------
[[nodiscard]] __host__ __device__ static inline double hd_log(const double x) {
#if defined(__CUDA_ARCH__)
    return ::log(x);
#else
    return std::log(x);
#endif
}

[[nodiscard]] __host__ __device__ static inline double hd_sqrt(const double x) {
#if defined(__CUDA_ARCH__)
    return ::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

[[nodiscard]] __host__ __device__ static inline double hd_max(const double a,
                                                              const double b) noexcept {
    return (a > b) ? a : b;
}

[[nodiscard]] __host__ __device__ static inline double hd_min(const double a,
                                                              const double b) noexcept {
    return (a < b) ? a : b;
}

[[nodiscard]] __host__ __device__ static inline double hd_clamp(const double x, const double lo,
                                                                const double hi) noexcept {
    return hd_min(hi, hd_max(lo, x));
}

/**
 * @brief Linear stopping power (dE/dx) for a charged ion in a material using the PDG Bethe
 * equation.
 *
 * Implements the PDG "Bethe equation" for heavy charged particles, including:
 *  - W_max from PDG Eq. (34.4)
 *  - stopping-power bracket from PDG Eq. (34.5), including the density-effect term -delta/2
 *
 * Returns *linear stopping power* in MeV/cm:
 *   (dE/dx)_linear = rho * (dE/dx)_mass
 *
 * Assumes that input values have already been checked as valid.
 *
 * @param projectile_velocity_ms
 *      Projectile velocity in metres per second.
 * @param projectile_atomic_number
 *      Projectile charge number z (number of protons in the ion).
 * @param projectile_atomic_mass_mev
 *      Projectile rest mass energy Mc^2 in MeV.
 * @param target_atomic_number
 *      Atomic number Z of the target material.
 * @param target_atomic_mass_g_mol
 *      Atomic mass A of the target material in g/mol.
 * @param target_density_g_cm3
 *      Target density rho in g/cm^3.
 * @param mean_excitation_energy_mev
 *      Mean excitation energy I in MeV.
 * @param density_effect_delta
 *      Density-effect correction delta(beta*gamma) (dimensionless). Use 0 if not applying.
 *
 * @return
 *      Linear stopping power dE/dx in MeV/cm.
 *
 * @warning
 *      This routine does not validate inputs. In particular, beta must be in (0, 1).
 *      This implementation clamps beta to avoid divide-by-zero and gamma overflow; that changes
 * physics.
 */
[[nodiscard]] __host__ __device__ static inline double stopping_power(
    const double projectile_velocity_ms, const int projectile_atomic_number,
    const double projectile_atomic_mass_mev, const int target_atomic_number,
    const double target_atomic_mass_g_mol, const double target_density_g_cm3,
    const double mean_excitation_energy_mev, const double density_effect_delta) {
    // Fundamental constants (PDG)
    static constexpr double SPEED_OF_LIGHT_MS = 299792458.0;    // [m/s]
    static constexpr double ELECTRON_MASS_MEV = 0.51099895000;  // [MeV]
    static constexpr double BETHE_CONSTANT_K = 0.307075;        // [MeV·cm^2/mol]
    static constexpr auto SMALL_VALUE{1.0e-9};

    // Relativistic kinematics
    const double beta_raw = projectile_velocity_ms / SPEED_OF_LIGHT_MS;
    const double beta = hd_clamp(beta_raw, SMALL_VALUE, 0.99999);  // clamped to avoid errors
    const double beta2 = beta * beta;

    const double inv_one_minus_beta2 = 1.0 / (1.0 - beta2);
    const double gamma2 = hd_max(0.0, inv_one_minus_beta2);
    const double gamma = hd_sqrt(gamma2);

    // Maximum energy transfer W_max (PDG Eq. 34.4)
    const double denom_mass = hd_max(SMALL_VALUE, projectile_atomic_mass_mev);
    const double electron_to_projectile_mass = ELECTRON_MASS_MEV / denom_mass;

    const double w_max_numerator = 2.0 * ELECTRON_MASS_MEV * beta2 * gamma2;

    const double w_max_denominator_raw =
        1.0 + 2.0 * gamma * electron_to_projectile_mass +
        (electron_to_projectile_mass * electron_to_projectile_mass);

    const double w_max_denominator = hd_max(w_max_denominator_raw, SMALL_VALUE);
    const double w_max_mev = w_max_numerator / w_max_denominator;

    // Logarithmic argument (PDG Eq. 34.5)
    const double mean_excitation_energy2_mev2 =
        mean_excitation_energy_mev * mean_excitation_energy_mev;

    const double log_arg_num = (2.0 * ELECTRON_MASS_MEV * beta2 * gamma2 * w_max_mev);
    const double log_arg_den = hd_max(SMALL_VALUE, mean_excitation_energy2_mev2);
    const double log_argument = hd_max(log_arg_num / log_arg_den, SMALL_VALUE);

    // Square-bracketed term (PDG Eq. 34.5 + density effect)
    const double bracket = 0.5 * hd_log(log_argument) - beta2 - 0.5 * density_effect_delta;

    // Mass stopping power [MeV·cm^2/g] and linear stopping power [MeV/cm]
    const double projectile_charge = static_cast<double>(projectile_atomic_number);
    const double projectile_charge2 = projectile_charge * projectile_charge;

    const double z_over_a =
        static_cast<double>(target_atomic_number) / hd_max(SMALL_VALUE, target_atomic_mass_g_mol);
    const double prefactor_mass = BETHE_CONSTANT_K * projectile_charge2 * z_over_a / beta2;

    const double mass_stopping_power_mev_cm2_per_g = prefactor_mass * bracket;
    const double linear_stopping_power_mev_per_cm =
        target_density_g_cm3 * mass_stopping_power_mev_cm2_per_g;

    return linear_stopping_power_mev_per_cm;
}

/**
 * @brief Compute stopping power for an array of projectile velocities (serial).
 *
 * @param velocity_array Projectile velocities in m/s.
 * @param results Output array (must be pre-sized to match velocity_array).
 *
 * @warning
 *      This routine does not validate sizes; callers must ensure `results.size() ==
 * velocity_array.size()`.
 */
static inline void serial_task(const std::vector<double>& velocity_array,
                               std::vector<double>& results) {
    static constexpr auto PROJECTILE_ATOMIC_NUMBER{1};
    static constexpr auto PROJECTILE_ATOMIC_MASS_MEV{938.2720813};

    static constexpr auto TARGET_ATOMIC_NUMBER{26};
    static constexpr auto TARGET_ATOMIC_MASS_G_MOL{55.845};
    static constexpr auto TARGET_DENSITY_G_CM3{7.874};

    static constexpr auto MEAN_EXCITATION_ENERGY_MEV{286.0e-6};
    static constexpr auto DENSITY_EFFECT_DELTA{0.0};

    const auto n{std::size_t(velocity_array.size())};

    for (auto i = std::size_t(0); i < n; i++) {
        results[i] =
            stopping_power(velocity_array[i], PROJECTILE_ATOMIC_NUMBER, PROJECTILE_ATOMIC_MASS_MEV,
                           TARGET_ATOMIC_NUMBER, TARGET_ATOMIC_MASS_G_MOL, TARGET_DENSITY_G_CM3,
                           MEAN_EXCITATION_ENERGY_MEV, DENSITY_EFFECT_DELTA);
    }
}

// -----------------------------
// CUDA kernel
// -----------------------------
__global__ void stopping_power_kernel(const std::size_t n,
                                      const double* __restrict__ velocity_device,
                                      double* __restrict__ stopping_power_device) {
    static constexpr int PROJECTILE_ATOMIC_NUMBER = 1;
    static constexpr double PROJECTILE_ATOMIC_MASS_MEV = 938.2720813;

    static constexpr int TARGET_ATOMIC_NUMBER = 26;
    static constexpr double TARGET_ATOMIC_MASS_G_MOL = 55.845;
    static constexpr double TARGET_DENSITY_G_CM3 = 7.874;

    static constexpr double MEAN_EXCITATION_ENERGY_MEV = 286.0e-6;
    static constexpr double DENSITY_EFFECT_DELTA = 0.0;

    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t i = tid; i < n; i += stride) {
        stopping_power_device[i] =
            stopping_power(velocity_device[i], PROJECTILE_ATOMIC_NUMBER, PROJECTILE_ATOMIC_MASS_MEV,
                           TARGET_ATOMIC_NUMBER, TARGET_ATOMIC_MASS_G_MOL, TARGET_DENSITY_G_CM3,
                           MEAN_EXCITATION_ENERGY_MEV, DENSITY_EFFECT_DELTA);
    }
}

static inline void launch_cuda_task(const std::size_t n, const double* velocity_device,
                                    double* stopping_power_device) {
    constexpr int block_size = 256;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const int blocks_for_n = static_cast<int>((n + static_cast<std::size_t>(block_size) - 1) /
                                              static_cast<std::size_t>(block_size));

    int blocks = prop.multiProcessorCount * 8;
    blocks = std::min(blocks, blocks_for_n);
    blocks = std::max(blocks, 1);

    stopping_power_kernel<<<blocks, block_size>>>(n, velocity_device, stopping_power_device);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit vec_size\n";
        return 1;
    }

    const auto test_time_s{std::atof(argv[1])};
    const auto n_raw{std::atoll(argv[2])};

    if (n_raw <= 0) {
        std::cerr << "vec_size must be a positive integer\n";
        return 1;
    }

    const auto n{static_cast<std::size_t>(n_raw)};

    // Setup timing start
    const auto t0{std::chrono::steady_clock::now()};

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cerr << "Using CUDA device: " << prop.name << "\n";

    // Host buffers (pinned for faster H2D/D2H, similar “host visible” intent)
    double* velocity_host = nullptr;
    double* stopping_power_host = nullptr;

    CUDA_CHECK(cudaMallocHost(&velocity_host, sizeof(double) * n));
    CUDA_CHECK(cudaMallocHost(&stopping_power_host, sizeof(double) * n));

    // Device buffers
    double* velocity_device = nullptr;
    double* stopping_power_device = nullptr;

    CUDA_CHECK(cudaMalloc(&velocity_device, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&stopping_power_device, sizeof(double) * n));

    // Fill input once (host writes shared memory)
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(1.0e7, 1.0e8);

    for (auto i = std::size_t(0); i < n; i++) {
        velocity_host[i] = dist(rng);
    }

    auto expected_value{0.0};

    // Expected value (serial reference)
    {
        auto velocity_host_vec{std::vector<double>(velocity_host, velocity_host + n)};
        auto stopping_power_host_vec{std::vector<double>(n)};
        serial_task(velocity_host_vec, stopping_power_host_vec);
        expected_value = helper::check_sum(stopping_power_host_vec);
        std::cout << "Serial computed expected value: " << expected_value << '\n';
    }

    CUDA_CHECK(
        cudaMemcpy(velocity_device, velocity_host, sizeof(double) * n, cudaMemcpyHostToDevice));

    // Calc timing start
    const auto t1{std::chrono::steady_clock::now()};
    const auto deadline{t1 + std::chrono::duration<double>(test_time_s)};
    auto iters{std::uint64_t(0)};

    // Run as many iterations as possible
    do {
        launch_cuda_task(n, velocity_device, stopping_power_device);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);

    // Ensure last submitted kernel finished before reading result
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto t2{std::chrono::steady_clock::now()};

    CUDA_CHECK(cudaMemcpy(stopping_power_host, stopping_power_device, sizeof(double) * n,
                          cudaMemcpyDeviceToHost));

    auto stopping_power_values = std::vector<double>(stopping_power_host, stopping_power_host + n);

    // Sum stopping_power on host for comparison
    auto calculated_value{0.0};
    for (auto i = std::size_t(0); i < n; i++) {
        calculated_value += stopping_power_host[i];
    }

    // Free
    CUDA_CHECK(cudaFree(stopping_power_device));
    CUDA_CHECK(cudaFree(velocity_device));
    CUDA_CHECK(cudaFreeHost(stopping_power_host));
    CUDA_CHECK(cudaFreeHost(velocity_host));

    const auto t3{std::chrono::steady_clock::now()};

    const auto time_setup_s{std::chrono::duration<double>(t1 - t0).count()};
    const auto time_calc_s{std::chrono::duration<double>(t2 - t1).count()};
    const auto time_cleanup_s{std::chrono::duration<double>(t3 - t2).count()};
    const auto time_total_s{std::chrono::duration<double>(t3 - t0).count()};
    const auto time_per_iteration_s{(iters > 0) ? (time_calc_s / static_cast<double>(iters)) : 0.0};

    const auto passed_check{(std::abs(calculated_value - expected_value) < 1.0e-6)};

    const auto method{std::string("Parallel CUDA")};
    const auto comments{std::string("stopping_power")};

    // Output
    {
        const std::string base_file_name = "results/parallel_cuda";
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = "Bethe-Bloch Stopping Power";
        j["comments"] = comments;
        j["threads"] = 1;
        j["device"] = "GPU";

        // Iteration/timing
        j["test_time_seconds"] = test_time_s;
        j["iterations"] = iters;
        j["time_per_iteration"] = time_per_iteration_s;
        j["time_setup"] = time_setup_s;
        j["time_calc"] = time_calc_s;
        j["time_cleanup"] = time_cleanup_s;
        j["time_total"] = time_total_s;

        // Values
        j["expected_value"] = helper::to_string_precise(expected_value);
        j["calculated_value"] = helper::to_string_precise(calculated_value);
        ;
        j["difference"] = helper::to_string_precise(expected_value - calculated_value);
        j["passed_check"] = passed_check;
        j["values"] = helper::to_string_precise_vector(stopping_power_values);

        // Memory
        // j["max_rss_kb"] = helper::max_rss_kb();

        std::ofstream out(json_file);
        if (!out) {
            throw std::runtime_error("Failed to open output JSON file.");
        }

        // Pretty-print. Use `out << j;` if you want compact.
        out << std::setw(2) << j << '\n';
    }

    return 0;
}
