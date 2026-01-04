// serial.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <limits>
#include <omp.h>
#include <random>

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
 * References:
 *  - Particle Data Group (PDG), "Passage of Particles Through Matter" (e.g. 2022 edition)
 *  - H. Bethe, Ann. Phys. 5, 325–400 (1930)
 *  - F. Bloch, Ann. Phys. 16, 285–320 (1933)
 *  - ICRU Report 49, Stopping Powers and Ranges for Protons and Alpha Particles
 *
 * Assumes that input values have already been checked as valid.
 *
 * @param projectile_velocity_ms
 *      Projectile velocity in metres per second.
 *
 * @param projectile_atomic_number
 *      Projectile charge number z (number of protons in the ion).
 *
 * @param projectile_atomic_mass
 *      Projectile rest mass energy Mc^2 in MeV.
 *
 * @param target_atomic_number
 *      Atomic number Z of the target material.
 *
 * @param target_atomic_mass
 *      Atomic mass A of the target material in g/mol.
 *
 * @param target_density_g_cm3
 *      Target density rho in g/cm^3.
 *
 * @param mean_excitation_energy_mev
 *      Mean excitation energy I in MeV.
 *
 * @param density_effect_delta
 *      Density-effect correction delta(beta*gamma) (dimensionless). Use 0 if not applying.
 *
 * @param shell_correction_c_over_z
 *      Shell correction term C/Z (dimensionless). Use 0 if not applying.
 *      (Included in square brackets as -C/Z.)
 *
 * @param include_spin_half_correction
 *      If true, includes the PDG spin-1/2 correction term +(W_max/E)^2/4 in the square brackets.
 *
 * @return
 *      Linear stopping power dE/dx in MeV/cm.
 */
static inline double stopping_power(
    const double projectile_velocity_ms,
    const int    projectile_atomic_number,
    const double projectile_atomic_mass,
    const int    target_atomic_number,
    const double target_atomic_mass,
    const double target_density_g_cm3,
    const double mean_excitation_energy_mev,
    const double density_effect_delta,
    const double shell_correction_c_over_z,
    const bool   include_spin_half_correction
)
{
    // ----------------------------
    // Fundamental constants (PDG Table values)
    // ----------------------------
    static constexpr double speed_of_light = 299792458.0;   ///< [m/s]
    static constexpr double electron_mass  = 0.51099895000; ///< electron mass energy [MeV]
    static constexpr double bethe_constant = 0.307075;      ///< K = 4π N_A r_e^2 m_e c^2 [MeV·cm^2/mol]

    // ----------------------------
    // Relativistic kinematics
    // ----------------------------
    const double beta  = projectile_velocity_ms / speed_of_light;
    const double beta2 = beta * beta;

    const double inv_one_minus_beta2 = 1.0 / (1.0 - beta2);
    const double gamma2 = inv_one_minus_beta2;
    const double gamma  = std::sqrt(gamma2);

    // Total energy E = gamma * M c^2  [MeV]
    const double total_energy_mev = gamma * projectile_atomic_mass;

    // ----------------------------
    // Maximum energy transfer W_max (PDG Eq. 34.4)
    // ----------------------------
    const double electron_to_projectile_mass = electron_mass / projectile_atomic_mass;

    const double w_max_numerator   = 2.0 * electron_mass * beta2 * gamma2;
    const double w_max_denominator = std::max(
        1.0
      + 2.0 * gamma * electron_to_projectile_mass
      + (electron_to_projectile_mass * electron_to_projectile_mass),
        1.0e-12
    );

    const double w_max = w_max_numerator / w_max_denominator; ///< [MeV]

    // ----------------------------
    // Logarithmic argument (PDG Eq. 34.5)
    // ----------------------------
    const double i2 = mean_excitation_energy_mev * mean_excitation_energy_mev;

    const double log_argument = std::max(
        (2.0 * electron_mass * beta2 * gamma2 * w_max) / i2,
        1.0
    );

    // ----------------------------
    // Square-bracketed term (PDG Eq. 34.5 + optional corrections)
    // ----------------------------
    double bracket =
        0.5 * std::log(log_argument)
      - beta2
      - 0.5 * density_effect_delta
      - shell_correction_c_over_z;

    if (include_spin_half_correction)
    {
        const double w_over_e = w_max / std::max(total_energy_mev, 1.0e-12);
        bracket += 0.25 * (w_over_e * w_over_e); // +(W_max/E)^2 / 4
    }

    // ----------------------------
    // Mass stopping power [MeV·cm^2/g] and linear stopping power [MeV/cm]
    // ----------------------------
    const double projectile_charge_squared =
        static_cast<double>(projectile_atomic_number)
      * static_cast<double>(projectile_atomic_number);

    const double z_over_a =
        static_cast<double>(target_atomic_number) / target_atomic_mass;

    const double prefactor_mass = bethe_constant * projectile_charge_squared * z_over_a / beta2;

    const double mass_stopping_power_mev_cm2_per_g = prefactor_mass * bracket;
    const double linear_stopping_power_mev_per_cm  = target_density_g_cm3 * mass_stopping_power_mev_cm2_per_g;

    return linear_stopping_power_mev_per_cm;
}







void serial_task(const std::vector<double>& velocity_array, std::vector<double>& results)
{
    for(size_t i=0; i<velocity_array.size(); i++)
    {
        results[i] = stopping_power(
            velocity_array[i],
            1,
            1.008,
            26,
            55.845,
            7.874,
            286.0,
            0.0,
            0.0,
            false
        );
    }

}



// Check the sum of an array (serial)
double check_sum(const std::vector<double>& numbers_c)
{
    double sum {};
    for(const auto& number : numbers_c)
    {
        sum = sum + number;
    }
    return sum;
}


int main(int argc, char** argv) 
{

    // Must have 3 arguments
    if (argc < 3) 
    {
        std::cerr << "Usage: " << argv[0] << " time_limit   vec_size\n";
        return 1;
    }

    // Read in test_time and size of vector
    double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);


    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);    // [0.0, 1.0)

    // Vector of numbers
    std::vector<double> velocity_array;
    velocity_array.reserve(N);

    // Populate vectors
    for (std::size_t i = 0; i < N; ++i) 
    {
        velocity_array.emplace_back(1.0e6 * dist(rng));
    }

    double expected_value {};

    // Expected value
    {
        std::vector<double> stopping_power(N);
        serial_task(velocity_array, stopping_power);
        expected_value = check_sum(stopping_power);
        std::cout << "Serial computed expected value:  " << expected_value << std::endl;
    }

  
    // ======= Calculation Starts ========

    // Setup
    auto t0 = std::chrono::steady_clock::now();


    // Do calculation
    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    int sum {};

    
    // Do as many times as possible before time runs out
    std::vector<double> stopping_power(N);
    do 
    {
        serial_task(velocity_array, stopping_power);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);

    // Clean up
    auto t2 = std::chrono::steady_clock::now();


    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========

    // Check
    double calculated_value = check_sum(stopping_power);
   
    double time_setup = std::chrono::duration<double>(t1 - t0).count();
    double time_calc = std::chrono::duration<double>(t2 - t1).count();
    double time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    double time_total = std::chrono::duration<double>(t3 - t0).count();
    double time_per_iteration = time_calc / iters;


    std::string method {"Serial"};
    std::string comments {"stopping_power"};
    bool passed_check = std::abs(calculated_value - expected_value) < 1.0e-9;

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
              << comments << "" 
              << std::endl;

    
    
    return 0;
}