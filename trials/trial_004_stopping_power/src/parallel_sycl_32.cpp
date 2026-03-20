// sycl_stopping_power_usm.cpp

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include "json.hpp"
#include "helper.hpp"
#include "Error.hpp"

#include <sycl/sycl.hpp>



/**
 * @brief Log function usable on host and SYCL device.
 *
 * @param x Input value.
 * @return Natural logarithm of x.
 */
[[nodiscard]]
static inline float sycl_compatible_log(const float x)
{
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::log(x);
#else
    return std::log(x);
#endif
}



/**
 * @brief Square-root function usable on host and SYCL device.
 *
 * @param x Input value.
 * @return Square root of x.
 */
[[nodiscard]]
static inline float sycl_compatible_sqrt(const float x)
{
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::sqrt(x);
#else
    return std::sqrt(x);
#endif
}



/**
 * @brief Linear stopping power (dE/dx) for a charged ion in a material using the PDG Bethe equation.
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
 *      This implementation clamps beta to avoid divide-by-zero and gamma overflow; that changes physics.
 */
[[nodiscard]]
static inline float stopping_power(
    const float projectile_velocity_ms,
    const int projectile_atomic_number,
    const float projectile_atomic_mass_mev,
    const int target_atomic_number,
    const float target_atomic_mass_g_mol,
    const float target_density_g_cm3,
    const float mean_excitation_energy_mev,
    const float density_effect_delta)
{
    // Fundamental constants (PDG)
    static constexpr auto SPEED_OF_LIGHT_MS {299792458.0f};    ///< [m/s]
    static constexpr auto ELECTRON_MASS_MEV {0.51099895000f};  ///< [MeV]
    static constexpr auto BETHE_CONSTANT_K  {0.307075f};       ///< [MeV·cm^2/mol]
    static constexpr auto SMALL_VALUE  {1.0e-9f};

    // Relativistic kinematics
    const auto beta_raw {projectile_velocity_ms / SPEED_OF_LIGHT_MS};
    const auto beta {std::clamp(beta_raw, SMALL_VALUE, 0.99999f)};            // Clamped to sensible values to avoid errors 
    const auto beta2 {beta * beta};

    const auto inv_one_minus_beta2 {1.0f / (1.0f - beta2)};
    const auto gamma2 {std::max(0.0f, inv_one_minus_beta2)};
    const auto gamma {sycl_compatible_sqrt(gamma2)};

    // Maximum energy transfer W_max (PDG Eq. 34.4)
    const auto electron_to_projectile_mass {ELECTRON_MASS_MEV / std::max(SMALL_VALUE, projectile_atomic_mass_mev)};

    const auto w_max_numerator {2.0f * ELECTRON_MASS_MEV * beta2 * gamma2};
    const auto w_max_denominator = std::max(
        1.0f
      + 2.0f * gamma * electron_to_projectile_mass
      + (electron_to_projectile_mass * electron_to_projectile_mass),
        SMALL_VALUE);

    const auto w_max_mev {w_max_numerator / w_max_denominator};

    // Logarithmic argument (PDG Eq. 34.5)
    const auto mean_excitation_energy2_mev2 {mean_excitation_energy_mev * mean_excitation_energy_mev};

    const auto log_argument = std::max(
        (2.0f * ELECTRON_MASS_MEV * beta2 * gamma2 * w_max_mev) / std::max(SMALL_VALUE, mean_excitation_energy2_mev2),
        SMALL_VALUE);

    // Square-bracketed term (PDG Eq. 34.5 + optional corrections)
    const auto bracket =
        0.5f * sycl_compatible_log(log_argument)
      - beta2
      - 0.5f * density_effect_delta;

    // Mass stopping power [MeV·cm^2/g] and linear stopping power [MeV/cm]
    const auto projectile_charge {static_cast<float>(projectile_atomic_number)};
    const auto projectile_charge2 {projectile_charge * projectile_charge};

    const auto z_over_a {static_cast<float>(target_atomic_number) / std::max(SMALL_VALUE, target_atomic_mass_g_mol)};
    const auto prefactor_mass {BETHE_CONSTANT_K * projectile_charge2 * z_over_a / beta2};

    const auto mass_stopping_power_mev_cm2_per_g {prefactor_mass * bracket};
    const auto linear_stopping_power_mev_per_cm {target_density_g_cm3 * mass_stopping_power_mev_cm2_per_g};

    return linear_stopping_power_mev_per_cm;
}



/**
 * @brief Compute stopping power for an array of projectile velocities (serial).
 *
 * @param velocity_array Projectile velocities in m/s.
 * @param results Output array (must be pre-sized to match velocity_array).
 *
 * @warning
 *      This routine does not validate sizes; callers must ensure `results.size() == velocity_array.size()`.
 */
static inline void serial_task(
    const std::vector<float>& velocity_array,
    std::vector<float>& results)
{
    static constexpr auto PROJECTILE_ATOMIC_NUMBER {1};
    static constexpr auto PROJECTILE_ATOMIC_MASS_MEV {938.2720813f};

    static constexpr auto TARGET_ATOMIC_NUMBER {26};
    static constexpr auto TARGET_ATOMIC_MASS_G_MOL {55.845f};
    static constexpr auto TARGET_DENSITY_G_CM3 {7.874f};

    static constexpr auto MEAN_EXCITATION_ENERGY_MEV {286.0e-6f};
    static constexpr auto DENSITY_EFFECT_DELTA {0.0f};

    const auto n {std::size_t(velocity_array.size())};

    for (auto i = std::size_t(0); i < n; i++)
    {
        results[i] = stopping_power(
            velocity_array[i],
            PROJECTILE_ATOMIC_NUMBER,
            PROJECTILE_ATOMIC_MASS_MEV,
            TARGET_ATOMIC_NUMBER,
            TARGET_ATOMIC_MASS_G_MOL,
            TARGET_DENSITY_G_CM3,
            MEAN_EXCITATION_ENERGY_MEV,
            DENSITY_EFFECT_DELTA);
    }
}



/**
 * @brief Fill per-particle stopping power array on device (USM device allocations).
 *
 * @param queue SYCL queue.
 * @param n Number of elements.
 * @param velocity_device Device pointer to velocities (length n).
 * @param stopping_power_device Device pointer to outputs (length n).
 *
 * @return Event for the submitted kernel.
 */
[[nodiscard]]
static inline sycl::event sycl_task(
    sycl::queue& queue,
    const std::size_t n,
    const float* const velocity_device,
    float* const stopping_power_device)
{
    return queue.parallel_for(
        sycl::range<1>(n),
        [=](sycl::item<1> item)
        {
            static constexpr auto PROJECTILE_ATOMIC_NUMBER {1};
            static constexpr auto PROJECTILE_ATOMIC_MASS_MEV {938.2720813f};

            static constexpr auto TARGET_ATOMIC_NUMBER {26};
            static constexpr auto TARGET_ATOMIC_MASS_G_MOL {55.845f};
            static constexpr auto TARGET_DENSITY_G_CM3 {7.874f};

            static constexpr auto MEAN_EXCITATION_ENERGY_MEV {286.0e-6f};
            static constexpr auto DENSITY_EFFECT_DELTA {0.0f};

            const auto i {item.get_linear_id()};

            stopping_power_device[i] = stopping_power(
                velocity_device[i],
                PROJECTILE_ATOMIC_NUMBER,
                PROJECTILE_ATOMIC_MASS_MEV,
                TARGET_ATOMIC_NUMBER,
                TARGET_ATOMIC_MASS_G_MOL,
                TARGET_DENSITY_G_CM3,
                MEAN_EXCITATION_ENERGY_MEV,
                DENSITY_EFFECT_DELTA);
        });
}



int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " time_limit vec_size\n";
        return 1;
    }

    const auto test_time_s {std::atof(argv[1])};
    const auto n_raw {std::atoll(argv[2])};

    if (n_raw <= 0)
    {
        std::cerr << "vec_size must be a positive integer\n";
        return 1;
    }

    const auto n {static_cast<std::size_t>(n_raw)};

    std::string_view device_string = "GPU";
    if (argc >= 5)
    {
        device_string = argv[4];

        if (device_string != "GPU" && device_string != "CPU")
        {
            THROW_INVALID_ARGUMENT("device must be GPU or CPU");
        }
    }

    // Setup timing start
    const auto t0 {std::chrono::steady_clock::now()};

    sycl::queue queue =
    (device_string == "CPU")
    ? sycl::queue{sycl::cpu_selector_v}
    : sycl::queue{sycl::gpu_selector_v};

    std::cerr << "Using device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    

    // Allocate USM: shared (host visible) + device (device-only)
    auto* velocity_host {sycl::malloc_shared<float>(n, queue)};
    auto* stopping_power_host {sycl::malloc_shared<float>(n, queue)};

    auto* velocity_device {sycl::malloc_device<float>(n, queue)};
    auto* stopping_power_device {sycl::malloc_device<float>(n, queue)};

    if ((velocity_host == nullptr)
     || (stopping_power_host == nullptr)
     || (velocity_device == nullptr)
     || (stopping_power_device == nullptr))
    {
        std::cerr << "Memory allocation failed\n";
        if (stopping_power_device != nullptr) sycl::free(stopping_power_device, queue);
        if (velocity_device != nullptr) sycl::free(velocity_device, queue);
        if (stopping_power_host != nullptr) sycl::free(stopping_power_host, queue);
        if (velocity_host != nullptr) sycl::free(velocity_host, queue);
        return 2;
    }

    // Fill input once (host writes shared memory)
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(1.0e7, 1.0e8); 

    for (auto i = std::size_t(0); i < n; i++)
    {
        velocity_host[i] = static_cast<float>(dist(rng));
    }

    auto expected_value {0.0};

    // Expected value (serial reference)
    {
        auto velocity_host_vec {std::vector<float>(velocity_host, velocity_host + n)};
        auto stopping_power_host_vec {std::vector<float>(n)};
        serial_task(velocity_host_vec, stopping_power_host_vec);
        expected_value = helper::check_sum(stopping_power_host_vec);
        std::cout << "Serial computed expected value: " << expected_value << '\n';
    }

    queue.memcpy(velocity_device, velocity_host, sizeof(float) * n).wait();

    // Calc timing start
    const auto t1 {std::chrono::steady_clock::now()};
    const auto deadline {t1 + std::chrono::duration<double>(test_time_s)};
    auto iters {std::uint64_t(0)};

    sycl::event last_event;

    // Run as many iterations as possible
    do
    {
        last_event = sycl_task(queue, n, velocity_device, stopping_power_device);
        iters++;
    }
    while (std::chrono::steady_clock::now() < deadline);

    // Ensure last submitted kernel finished before reading result
    last_event.wait();

    const auto t2 {std::chrono::steady_clock::now()};

    queue.memcpy(stopping_power_host, stopping_power_device, sizeof(float) * n).wait();

    auto stopping_power_values = std::vector<float>(stopping_power_host, stopping_power_host + n);

    // Sum stopping_power on host for comparison
    auto calculated_value {0.0};
    for (auto i = std::size_t(0); i < n; i++)
    {
        calculated_value += stopping_power_values[i];
    }

    // Free USM
    sycl::free(stopping_power_device, queue);
    sycl::free(stopping_power_host, queue);
    sycl::free(velocity_device, queue);
    sycl::free(velocity_host, queue);

    const auto t3 {std::chrono::steady_clock::now()};

    const auto time_setup_s {std::chrono::duration<double>(t1 - t0).count()};
    const auto time_calc_s {std::chrono::duration<double>(t2 - t1).count()};
    const auto time_cleanup_s {std::chrono::duration<double>(t3 - t2).count()};
    const auto time_total_s {std::chrono::duration<double>(t3 - t0).count()};
    const auto time_per_iteration_s {(iters > 0) ? (time_calc_s / static_cast<double>(iters)) : 0.0};

    const auto passed_check {(std::abs(calculated_value - expected_value) < 1.0e-6)};

    const auto method {std::string("Parallel Sycl 32")};
    const auto comments {std::string("stopping_power")};

    // Output
    {

        const std::string base_file_name = "results/parallel_sycl_32";
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        // Cast to double for output
        auto stopping_power_values_out {std::vector<double>{}};
        stopping_power_values_out.reserve(stopping_power_values.size());
        for (auto i = std::size_t(0); i < stopping_power_values.size(); i++)
        {
            stopping_power_values_out.emplace_back(static_cast<double>(stopping_power_values[i]));
        }

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
        j["difference"] = helper::to_string_precise(expected_value - calculated_value);
        j["passed_check"] = passed_check;
        j["values"] = helper::to_string_precise_vector(stopping_power_values_out);

        // Memory
        //j["max_rss_kb"] = helper::max_rss_kb();

        std::ofstream out(json_file);
        if (!out)
        {
            throw std::runtime_error("Failed to open output JSON file.");
        }

        // Pretty-print. Use `out << j;` if you want compact.
        out << std::setw(2) << j << '\n';
    }

    return 0;
}
