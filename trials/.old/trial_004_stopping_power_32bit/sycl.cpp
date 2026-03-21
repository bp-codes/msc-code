// sycl_stopping_power_usm.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#if defined(__cpp_lib_format)
    #include <format>
#endif


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
 *  - optional spin-1/2 correction term +(W_max/E)^2/4 (PDG footnote)
 *  - optional shell correction term -C/Z (PDG low-energy discussion)
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
 * @param shell_correction_c_over_z
 *      Shell correction term C/Z (dimensionless). Use 0 if not applying.
 *      (Included in square brackets as -C/Z.)
 * @param include_spin_half_correction
 *      If true, includes the PDG spin-1/2 correction term +(W_max/E)^2/4 in the square brackets.
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
    const float density_effect_delta,
    const float shell_correction_c_over_z,
    const bool include_spin_half_correction)
{
    // Fundamental constants (PDG)
    static constexpr float SPEED_OF_LIGHT_MS {299792458.0f};    ///< [m/s]
    static constexpr float ELECTRON_MASS_MEV {0.51099895f};     ///< [MeV]
    static constexpr float BETHE_CONSTANT_K  {0.307075f};       ///< [MeV·cm^2/mol]

    // Relativistic kinematics
    const auto beta_raw {projectile_velocity_ms / SPEED_OF_LIGHT_MS};
    const auto beta {std::clamp(beta_raw, 1.0e-9f, 0.99999f)};
    const auto beta2 {beta * beta};

    const auto inv_one_minus_beta2 {1.0f / (1.0f - beta2)};
    const auto gamma2 {inv_one_minus_beta2};
    const auto gamma {sycl_compatible_sqrt(gamma2)};

    // Total energy E = gamma * M c^2 [MeV]
    const auto total_energy_mev {gamma * projectile_atomic_mass_mev};

    // Maximum energy transfer W_max (PDG Eq. 34.4)
    const auto electron_to_projectile_mass {ELECTRON_MASS_MEV / projectile_atomic_mass_mev};

    const auto w_max_numerator {2.0f * ELECTRON_MASS_MEV * beta2 * gamma2};
    const auto w_max_denominator = std::max(
        1.0f
      + 2.0f * gamma * electron_to_projectile_mass
      + (electron_to_projectile_mass * electron_to_projectile_mass),
        1.0e-12f);

    const auto w_max_mev {w_max_numerator / w_max_denominator};

    // Logarithmic argument (PDG Eq. 34.5)
    const auto mean_excitation_energy2_mev2
        {mean_excitation_energy_mev * mean_excitation_energy_mev};

    const auto log_argument = std::max(
        (2.0f * ELECTRON_MASS_MEV * beta2 * gamma2 * w_max_mev)
        / mean_excitation_energy2_mev2,
        1.0f);

    // Square-bracketed term (PDG Eq. 34.5 + optional corrections)
    auto bracket =
        0.5f * sycl_compatible_log(log_argument)
      - beta2
      - 0.5f * density_effect_delta
      - shell_correction_c_over_z;

    if (include_spin_half_correction)
    {
        const auto w_over_e
            {w_max_mev / std::max(total_energy_mev, 1.0e-12f)};
        bracket += 0.25f * (w_over_e * w_over_e); // +(W_max/E)^2 / 4
    }

    // Mass stopping power [MeV·cm^2/g] and linear stopping power [MeV/cm]
    const auto projectile_charge {static_cast<float>(projectile_atomic_number)};
    const auto projectile_charge2 {projectile_charge * projectile_charge};

    const auto z_over_a
        {static_cast<float>(target_atomic_number) / target_atomic_mass_g_mol};

    const auto prefactor_mass
        {BETHE_CONSTANT_K * projectile_charge2 * z_over_a / beta2};

    const auto mass_stopping_power_mev_cm2_per_g
        {prefactor_mass * bracket};

    const auto linear_stopping_power_mev_per_cm
        {target_density_g_cm3 * mass_stopping_power_mev_cm2_per_g};

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
    static constexpr int   PROJECTILE_ATOMIC_NUMBER {1};
    static constexpr float PROJECTILE_ATOMIC_MASS_MEV {938.2720813f};

    static constexpr int   TARGET_ATOMIC_NUMBER {26};
    static constexpr float TARGET_ATOMIC_MASS_G_MOL {55.845f};
    static constexpr float TARGET_DENSITY_G_CM3 {7.874f};

    static constexpr float MEAN_EXCITATION_ENERGY_MEV {286.0e-6f};
    static constexpr float DENSITY_EFFECT_DELTA {0.0f};
    static constexpr float SHELL_CORRECTION_C_OVER_Z {0.0f};
    static constexpr bool  INCLUDE_SPIN_HALF_CORRECTION {false};

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
            DENSITY_EFFECT_DELTA,
            SHELL_CORRECTION_C_OVER_Z,
            INCLUDE_SPIN_HALF_CORRECTION);
    }
}



/**
 * @brief Fill per-particle stopping power array on device (USM device allocations).
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
            static constexpr int   PROJECTILE_ATOMIC_NUMBER {1};
            static constexpr float PROJECTILE_ATOMIC_MASS_MEV {938.2720813f};

            static constexpr int   TARGET_ATOMIC_NUMBER {26};
            static constexpr float TARGET_ATOMIC_MASS_G_MOL {55.845f};
            static constexpr float TARGET_DENSITY_G_CM3 {7.874f};

            static constexpr float MEAN_EXCITATION_ENERGY_MEV {286.0e-6f};
            static constexpr float DENSITY_EFFECT_DELTA {0.0f};
            static constexpr float SHELL_CORRECTION_C_OVER_Z {0.0f};
            static constexpr bool  INCLUDE_SPIN_HALF_CORRECTION {false};

            const auto i {item.get_linear_id()};

            stopping_power_device[i] = stopping_power(
                velocity_device[i],
                PROJECTILE_ATOMIC_NUMBER,
                PROJECTILE_ATOMIC_MASS_MEV,
                TARGET_ATOMIC_NUMBER,
                TARGET_ATOMIC_MASS_G_MOL,
                TARGET_DENSITY_G_CM3,
                MEAN_EXCITATION_ENERGY_MEV,
                DENSITY_EFFECT_DELTA,
                SHELL_CORRECTION_C_OVER_Z,
                INCLUDE_SPIN_HALF_CORRECTION);
        });
}



/**
 * @brief Compute the sum of an array (serial).
 */
[[nodiscard]]
static inline float check_sum(const std::vector<float>& values) noexcept
{
    auto sum {0.0f};
    for (const auto value : values)
    {
        sum += value;
    }
    return sum;
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

    const auto t0 {std::chrono::steady_clock::now()};

    sycl::queue queue {sycl::default_selector_v};
    std::cerr << "Using device: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << '\n';

    auto* velocity_host {sycl::malloc_shared<float>(n, queue)};
    auto* stopping_power_host {sycl::malloc_shared<float>(n, queue)};
    auto* velocity_device {sycl::malloc_device<float>(n, queue)};
    auto* stopping_power_device {sycl::malloc_device<float>(n, queue)};

    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto i = std::size_t(0); i < n; i++)
    {
        velocity_host[i] = 1.0e6f * dist(rng);
    }

    float expected_value {0.0f};
    {
        auto v {std::vector<float>(velocity_host, velocity_host + n)};
        auto s {std::vector<float>(n)};
        serial_task(v, s);
        expected_value = check_sum(s);
    }

    queue.memcpy(velocity_device, velocity_host, sizeof(float) * n).wait();

    const auto t1 {std::chrono::steady_clock::now()};
    const auto deadline {t1 + std::chrono::duration<double>(test_time_s)};
    auto iters {std::uint64_t(0)};

    sycl::event last_event;
    do
    {
        last_event = sycl_task(queue, n, velocity_device, stopping_power_device);
        ++iters;
    }
    while (std::chrono::steady_clock::now() < deadline);

    last_event.wait();
    const auto t2 {std::chrono::steady_clock::now()};

    queue.memcpy(stopping_power_host, stopping_power_device, sizeof(float) * n).wait();

    float calculated_value {0.0f};
    for (auto i = std::size_t(0); i < n; i++)
    {
        calculated_value += stopping_power_host[i];
    }

    sycl::free(stopping_power_device, queue);
    sycl::free(stopping_power_host, queue);
    sycl::free(velocity_device, queue);
    sycl::free(velocity_host, queue);

    const auto t3 {std::chrono::steady_clock::now()};

    const auto time_setup_s   {std::chrono::duration<double>(t1 - t0).count()};
    const auto time_calc_s    {std::chrono::duration<double>(t2 - t1).count()};
    const auto time_cleanup_s {std::chrono::duration<double>(t3 - t2).count()};
    const auto time_total_s   {std::chrono::duration<double>(t3 - t0).count()};
    const auto time_per_iteration_s
        {(iters > 0) ? (time_calc_s / static_cast<double>(iters)) : 0.0};

    const auto passed_check {(std::abs(calculated_value - expected_value) < 1.0e-5f)};

    const auto method {std::string("SYCL_USM_array")};
    const auto comments {std::string("stopping_power")};

#if defined(__cpp_lib_format)
    std::cout << std::format(
        "{},{:.9g},{:.9g},{},{:.9e},{:.6f},{:.6f},{:.6f},{:.6f},{},{}\n",
        method,
        expected_value,
        calculated_value,
        iters,
        time_per_iteration_s,
        time_setup_s,
        time_calc_s,
        time_cleanup_s,
        time_total_s,
        passed_check,
        comments);
#else
    std::cout
        << method << ","
        << expected_value << ","
        << calculated_value << ","
        << iters << ","
        << time_per_iteration_s << ","
        << time_setup_s << ","
        << time_calc_s << ","
        << time_cleanup_s << ","
        << time_total_s << ","
        << passed_check << ","
        << comments
        << '\n';
#endif

    return 0;
}
