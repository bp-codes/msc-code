// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <format>
#include <algorithm>



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
    static constexpr float SPEED_OF_LIGHT_MS {299792458.0f};
    static constexpr float ELECTRON_MASS_MEV {0.51099895f};
    static constexpr float BETHE_CONSTANT_K  {0.307075f};

    // Relativistic kinematics
    const auto beta_raw {projectile_velocity_ms / SPEED_OF_LIGHT_MS};
    const auto beta {std::clamp(beta_raw, 1.0e-9f, 0.99999f)};
    const auto beta2 {beta * beta};

    const auto inv_one_minus_beta2 {1.0f / (1.0f - beta2)};
    const auto gamma2 {inv_one_minus_beta2};
    const auto gamma {std::sqrt(gamma2)};

    // Total energy E = gamma * M c^2 [MeV]
    const auto total_energy_mev {gamma * projectile_atomic_mass_mev};

    // Maximum energy transfer W_max
    const auto electron_to_projectile_mass {ELECTRON_MASS_MEV / projectile_atomic_mass_mev};

    const auto w_max_numerator {2.0f * ELECTRON_MASS_MEV * beta2 * gamma2};
    const auto w_max_denominator = std::max(
        1.0f
      + 2.0f * gamma * electron_to_projectile_mass
      + (electron_to_projectile_mass * electron_to_projectile_mass),
        1.0e-12f);

    const auto w_max_mev {w_max_numerator / w_max_denominator};

    // Logarithmic argument
    const auto mean_excitation_energy2_mev2
        {mean_excitation_energy_mev * mean_excitation_energy_mev};

    const auto log_argument = std::max(
        (2.0f * ELECTRON_MASS_MEV * beta2 * gamma2 * w_max_mev)
        / mean_excitation_energy2_mev2,
        1.0f);

    auto bracket =
        0.5f * std::log(log_argument)
      - beta2
      - 0.5f * density_effect_delta
      - shell_correction_c_over_z;

    if (include_spin_half_correction)
    {
        const auto w_over_e
            {w_max_mev / std::max(total_energy_mev, 1.0e-12f)};
        bracket += 0.25f * (w_over_e * w_over_e);
    }

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
    const auto n_raw {std::atoi(argv[2])};

    if (n_raw <= 0)
    {
        std::cerr << "vec_size must be a positive integer\n";
        return 1;
    }

    const auto n {static_cast<std::size_t>(n_raw)};

    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    auto velocity_array {std::vector<float>{}};
    velocity_array.reserve(n);

    for (auto i = std::size_t(0); i < n; i++)
    {
        velocity_array.emplace_back(1.0e6f * dist(rng));
    }

    auto expected_value {0.0f};

    {
        auto stopping_power_values {std::vector<float>(n)};
        serial_task(velocity_array, stopping_power_values);
        expected_value = check_sum(stopping_power_values);
        std::cout << "Serial computed expected value: "
                  << expected_value << '\n';
    }

    const auto t0 {std::chrono::steady_clock::now()};
    const auto t1 {std::chrono::steady_clock::now()};
    const auto deadline {t1 + std::chrono::duration<double>(test_time_s)};

    auto iters {std::uint64_t(0)};
    auto stopping_power_values {std::vector<float>(n)};

    do
    {
        serial_task(velocity_array, stopping_power_values);
        iters++;
    }
    while (std::chrono::steady_clock::now() < deadline);

    const auto t2 {std::chrono::steady_clock::now()};
    const auto t3 {std::chrono::steady_clock::now()};

    const auto calculated_value {check_sum(stopping_power_values)};

    const auto time_setup_s
        {std::chrono::duration<double>(t1 - t0).count()};
    const auto time_calc_s
        {std::chrono::duration<double>(t2 - t1).count()};
    const auto time_cleanup_s
        {std::chrono::duration<double>(t3 - t2).count()};
    const auto time_total_s
        {std::chrono::duration<double>(t3 - t0).count()};
    const auto time_per_iteration_s
        {time_calc_s / static_cast<double>(iters)};

    const auto passed_check
        {(std::abs(calculated_value - expected_value) < 1.0e-5f)};

    std::cout << std::format(
        "{},{:.9g},{:.9g},{},{:.9e},{:.6f},{:.6f},{:.6f},{:.6f},{},{}\n",
        "Serial",
        expected_value,
        calculated_value,
        iters,
        time_per_iteration_s,
        time_setup_s,
        time_calc_s,
        time_cleanup_s,
        time_total_s,
        passed_check,
        "stopping_power");

    return 0;
}
