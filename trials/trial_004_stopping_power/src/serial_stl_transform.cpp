// serial.cpp

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
#include "helper/helper.hpp"

#include <nlohmann/json.hpp>

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
[[nodiscard]]
static inline double stopping_power(
    const double projectile_velocity_ms, const int projectile_atomic_number,
    const double projectile_atomic_mass_mev, const int target_atomic_number,
    const double target_atomic_mass_g_mol, const double target_density_g_cm3,
    const double mean_excitation_energy_mev, const double density_effect_delta) {
    // Fundamental constants (PDG)
    static constexpr auto SPEED_OF_LIGHT_MS{299792458.0};    ///< [m/s]
    static constexpr auto ELECTRON_MASS_MEV{0.51099895000};  ///< [MeV]
    static constexpr auto BETHE_CONSTANT_K{0.307075};        ///< [MeV·cm^2/mol]
    static constexpr auto SMALL_VALUE{1.0e-9};

    // Relativistic kinematics
    const auto beta_raw{projectile_velocity_ms / SPEED_OF_LIGHT_MS};
    const auto beta{
        std::clamp(beta_raw, SMALL_VALUE, 0.99999)};  // Clamped to sensible values to avoid errors
    const auto beta2{beta * beta};

    const auto inv_one_minus_beta2{1.0 / (1.0 - beta2)};
    const auto gamma2{std::max(0.0, inv_one_minus_beta2)};
    const auto gamma{std::sqrt(gamma2)};

    // Maximum energy transfer W_max (PDG Eq. 34.4)
    const auto electron_to_projectile_mass{ELECTRON_MASS_MEV /
                                           std::max(SMALL_VALUE, projectile_atomic_mass_mev)};

    const auto w_max_numerator{2.0 * ELECTRON_MASS_MEV * beta2 * gamma2};
    const auto w_max_denominator =
        std::max(1.0 + 2.0 * gamma * electron_to_projectile_mass +
                     (electron_to_projectile_mass * electron_to_projectile_mass),
                 SMALL_VALUE);

    const auto w_max_mev{w_max_numerator / w_max_denominator};

    // Logarithmic argument (PDG Eq. 34.5)
    const auto mean_excitation_energy2_mev2{mean_excitation_energy_mev *
                                            mean_excitation_energy_mev};

    const auto log_argument = std::max((2.0 * ELECTRON_MASS_MEV * beta2 * gamma2 * w_max_mev) /
                                           std::max(SMALL_VALUE, mean_excitation_energy2_mev2),
                                       SMALL_VALUE);

    // Square-bracketed term (PDG Eq. 34.5 + optional corrections)
    const auto bracket = 0.5 * std::log(log_argument) - beta2 - 0.5 * density_effect_delta;

    // Mass stopping power [MeV·cm^2/g] and linear stopping power [MeV/cm]
    const auto projectile_charge{static_cast<double>(projectile_atomic_number)};
    const auto projectile_charge2{projectile_charge * projectile_charge};

    const auto z_over_a{static_cast<double>(target_atomic_number) /
                        std::max(SMALL_VALUE, target_atomic_mass_g_mol)};
    const auto prefactor_mass{BETHE_CONSTANT_K * projectile_charge2 * z_over_a / beta2};

    const auto mass_stopping_power_mev_cm2_per_g{prefactor_mass * bracket};
    const auto linear_stopping_power_mev_per_cm{target_density_g_cm3 *
                                                mass_stopping_power_mev_cm2_per_g};

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
static inline void task(const std::vector<double>& velocity_array,
                                   std::vector<double>& results) {
    // Parameters
    static constexpr auto PROJECTILE_ATOMIC_NUMBER{1};
    static constexpr auto PROJECTILE_ATOMIC_MASS_MEV{938.2720813};  // proton rest mass energy [MeV]

    static constexpr auto TARGET_ATOMIC_NUMBER{26};
    static constexpr auto TARGET_ATOMIC_MASS_G_MOL{55.845};
    static constexpr auto TARGET_DENSITY_G_CM3{7.874};

    static constexpr auto MEAN_EXCITATION_ENERGY_MEV{286.0e-6};  // 286 eV = 286e-6 MeV
    static constexpr auto DENSITY_EFFECT_DELTA{0.0};
    static constexpr auto SHELL_CORRECTION_C_OVER_Z{0.0};

    std::transform(velocity_array.begin(), velocity_array.end(), results.begin(),
                   [](const auto& velocity) {
                       return stopping_power(velocity, PROJECTILE_ATOMIC_NUMBER,
                                             PROJECTILE_ATOMIC_MASS_MEV, TARGET_ATOMIC_NUMBER,
                                             TARGET_ATOMIC_MASS_G_MOL, TARGET_DENSITY_G_CM3,
                                             MEAN_EXCITATION_ENERGY_MEV, DENSITY_EFFECT_DELTA);
                   });
}

int main(int argc, char** argv) {
    // Must have 3 arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit vec_size\n";
        return 1;
    }

    // Read in test_time and size of vector
    const auto test_time_s{std::atof(argv[1])};
    const auto n_raw{std::atoi(argv[2])};

    if (n_raw <= 0) {
        std::cerr << "vec_size must be a positive integer\n";
        return 1;
    }

    const auto n{static_cast<std::size_t>(n_raw)};

    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(1.0e7, 1.0e8);  // [0.0, 1.0)

    // Vector of numbers
    auto velocity_array{std::vector<double>{}};
    velocity_array.reserve(n);

    // Populate vectors
    std::generate_n(std::back_inserter(velocity_array), n, [&]() { return dist(rng); });

    // ======= Calculation Starts ========

    const auto t0{std::chrono::steady_clock::now()};

    // Do calculation
    const auto t1{std::chrono::steady_clock::now()};
    const auto deadline{t1 + std::chrono::duration<double>(test_time_s)};
    auto iters{std::uint64_t(0)};

    auto stopping_power_values{std::vector<double>(n)};

    // Do as many times as possible before time runs out
    do {
        task(velocity_array, stopping_power_values);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);

    const auto t2{std::chrono::steady_clock::now()};

    // Actual end time
    const auto t3{std::chrono::steady_clock::now()};

    // ======= Calculation Ends ========

    // Check
    const auto calculated_value{helper::check_sum(stopping_power_values)};

    const auto time_setup_s{std::chrono::duration<double>(t1 - t0).count()};
    const auto time_calc_s{std::chrono::duration<double>(t2 - t1).count()};
    const auto time_cleanup_s{std::chrono::duration<double>(t3 - t2).count()};
    const auto time_total_s{std::chrono::duration<double>(t3 - t0).count()};
    const auto time_per_iteration_s{time_calc_s / static_cast<double>(iters)};

    const auto method{std::string("Serial STL Transform")};
    const auto comments{std::string("stopping_power")};

    // Output
    {
        const std::string base_file_name = "results/serial_stl_transform";
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        // Metadata / identity
        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = "Bethe-Bloch Stopping Power";
        j["comments"] = comments;
        j["threads"] = 1;
        j["precision"] = "64";
        j["device"] = "CPU";

        // Iteration/timing
        j["test_time_seconds"] = test_time_s;
        j["iterations"] = iters;
        j["time_per_iteration"] = time_per_iteration_s;
        j["time_setup"] = time_setup_s;
        j["time_calc"] = time_calc_s;
        j["time_cleanup"] = time_cleanup_s;
        j["time_total"] = time_total_s;

        // Values
        j["calculated_value"] = helper::to_string_precise(calculated_value);
        j["values"] = helper::to_string_precise_vector(stopping_power_values);

        // Memory
        j["max_rss_kb"] = helper::max_rss_kb();

        std::ofstream out(json_file);
        if (!out) {
            throw std::runtime_error("Failed to open output JSON file.");
        }

        // Pretty-print. Use `out << j;` if you want compact.
        out << std::setw(2) << j << '\n';
    }

    return 0;
}
