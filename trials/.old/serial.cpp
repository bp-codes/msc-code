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
 * @brief Linear stopping power (dE/dx) for a charged ion in a material using Bethe–Bloch.
 *
 * Computes the relativistic Bethe–Bloch stopping power for a heavy charged projectile
 * traversing a homogeneous elemental target material.
 *
 * The returned quantity is the *linear stopping power*:
 *   dE/dx in MeV/cm
 * 
 * References:
 *  - H. Bethe, Ann. Phys. 5, 325–400 (1930)
 *  - F. Bloch, Ann. Phys. 16, 285–320 (1933)
 *  - Particle Data Group, "Passage of particles through matter",
 *    Prog. Theor. Exp. Phys. (latest edition)
 *  - ICRU Report 49, Stopping Powers and Ranges for Protons and Alpha Particles
 * https://pdg.lbl.gov/2017/reviews/rpp2017-rev-passage-particles-matter.pdf
 * 
 * Assumes that input values have already been checked as valid inputs.
 *
 * @note
 * - Valid for heavy charged particles (protons and ions).
 * - Not valid for electrons or positrons.
 * - Density-effect, shell corrections, and Barkas/Bloch terms are NOT included.
 * - At very low velocities (beta → 0), this expression diverges.
 *
 * @param projectile_velocity_ms
 *      Projectile velocity in metres per second.
 *
 * @param projectile_atomic_number
 *      Projectile charge number z (number of protons in the ion).
 *
 * @param projectile_atomic_mass
 *      Projectile rest mass energy in MeV.
 *
 * @param target_atomic_number
 *      Atomic number Z of the target material.
 *
 * @param target_atomic_mass
 *      Atomic mass A of the target material in g/mol.
 *
 * @return
 *      Linear stopping power dE/dx in MeV/cm.
 */
static inline double stopping_power(
    const double projectile_velocity_ms,
    const int    projectile_atomic_number,
    const double projectile_atomic_mass,
    const int    target_atomic_number,
    const double target_atomic_mass
)
{
    // ----------------------------
    // Fundamental constants
    // ----------------------------
    static constexpr double speed_of_light = 299792458.0;   ///< [m/s]
    static constexpr double electron_mass  = 0.51099895000; ///< [MeV]
    static constexpr double bethe_constant = 0.307075;      ///< [MeV·cm^2/mol]

    // ----------------------------
    // Mean excitation energy
    // ----------------------------
    // NOTE: Currently fixed (Fe). Must be generalized for arbitrary materials.
    static constexpr double mean_excitation_energy = 286.0e-6; ///< [MeV]

    // ----------------------------
    // Relativistic kinematics
    // ----------------------------
    const double beta  = projectile_velocity_ms / speed_of_light;
    const double beta2 = beta * beta;

    const double inv_one_minus_beta2 = 1.0 / (1.0 - beta2);
    const double gamma2 = inv_one_minus_beta2;
    const double gamma  = std::sqrt(gamma2);

    // ----------------------------
    // Maximum energy transfer
    // ----------------------------
    const double electron_to_projectile_mass = electron_mass / projectile_atomic_mass;

    const double numerator = 2.0 * electron_mass * beta2 * gamma2;

    const double denominator = std::max(
                                        1.0 + 2.0 * gamma * electron_to_projectile_mass
                                        + (electron_to_projectile_mass * electron_to_projectile_mass), 1.0e-9
                                        );
    

    const double t_max = numerator / denominator; ///< [MeV]

    // ----------------------------
    // Bethe–Bloch logarithmic term
    // ----------------------------
    const double log_argument = std::max((2.0 * electron_mass * beta2 * gamma2 * t_max) / (mean_excitation_energy * mean_excitation_energy), 1.0);

    // ----------------------------
    // Stopping power
    // ----------------------------
    const double projectile_charge_squared = static_cast<double>(projectile_atomic_number) * static_cast<double>(projectile_atomic_number);

    const double z_over_a = static_cast<double>(target_atomic_number) / target_atomic_mass;

    const double prefactor = bethe_constant * projectile_charge_squared * z_over_a / beta2;

    const double bracket = 0.5 * std::log(log_argument) - beta2;

    // Linear stopping power [MeV/cm]
    return prefactor * bracket;
}






void serial_task(const std::vector<double>& velocity_array, std::vector<double>& results)
{
    for(size_t i=0; i<velocity_array.size(); i++)
    {
        results[i] = stopping_power_fe(velocity_array[i]);
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