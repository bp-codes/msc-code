#ifndef CONFIGURATION_ENGINE_HPP
#define CONFIGURATION_ENGINE_HPP

/*********************************************************************************************************************************/
#include <omp.h>

#include <algorithm>     // std::max
#include <chrono>        // std::chrono::steady_clock
#include <cmath>         // std::sqrt
#include <cstddef>       // std::size_t
#include <filesystem>    // std::filesystem::path, create_directories, exists
#include <fstream>       // std::ofstream
#include <iomanip>       // std::fixed, std::setprecision
#include <ios>           // std::ios_base::openmode
#include <optional>      // std::optional
#include <random>        // std::mt19937, std::normal_distribution, std::random_device
#include <stdexcept>     // std::runtime_error
#include <system_error>  // std::error_code

#include "Helper/_helper.hpp"
#include "Maths/_maths.hpp"
#include "SimpleMD/Atom.hpp"
#include "SimpleMD/Configuration.hpp"
#include "SimpleMD/Morse.hpp"
#include "SimpleMD/Timer.hpp"

/*********************************************************************************************************************************/

namespace SimpleMD {

/**
 * @brief Utilities for operating on and updating a simulation Configuration.
 *
 * Provides helpers for perturbing/heating atom positions, building and updating
 * neighbour lists, and recording configurations to XYZ files.
 */
class ConfigurationEngine {
public:
    /**
     * @brief Heat the configuration to a target temperature for testing.
     *
     * Applies random perturbations to atom positions using a normal distribution.
     *
     * @param configuration Configuration to modify.
     * @param T_target Target temperature (currently unused).
     * @param kB Boltzmann constant (eV/K by default; currently unused).
     * @param seed Optional RNG seed for reproducibility.
     */
    static inline void heat(Configuration& configuration, double T_target,
                            double kB = 8.617333262145e-5,
                            std::optional<unsigned> seed = std::nullopt) {
        auto& atoms = configuration.get_atoms();
        const double sigma = configuration.get_heat();

        if (atoms.empty()) {
            return;
        }

        auto rng{std::mt19937{seed ? *seed : std::random_device{}()}};
        auto normal01{std::normal_distribution<double>{0.0, 1.0}};

        for (auto& atom : atoms) {
            Maths::Vec3 d_position{sigma * normal01(rng), sigma * normal01(rng),
                                   sigma * normal01(rng)};

            atom.set_position(atom.position + d_position);
        }
    }

    /**
     * @brief Apply a Gaussian positional perturbation to all atoms.
     *
     * @param n Scaling factor used in sigma adjustment.
     * @param sigma Base perturbation magnitude.
     * @param configuration Configuration to modify.
     */
    static inline void perturb(const std::size_t n, const double sigma,
                               Configuration& configuration) {
        auto& atoms = configuration.get_atoms();

        if (atoms.empty()) {
            return;
        }

        auto rng{std::mt19937{42}};
        auto normal01{std::normal_distribution<double>{0.0, 1.0}};

        for (auto& atom : atoms) {
            const double new_sigma = sigma * (1.0 / n);
            Maths::Vec3 d_position{new_sigma * normal01(rng), new_sigma * normal01(rng),
                                   new_sigma * normal01(rng)};

            atom.set_position(atom.position + d_position);
        }
    }

    /**
     * @brief Build the neighbour list for the current configuration.
     *
     * @param configuration Configuration whose neighbour list will be rebuilt.
     */
    static void make_neighbour_list(Configuration& configuration) {
        auto t0{std::chrono::steady_clock::now()};

        const auto& atoms = configuration.get_atoms();
        const double alat = configuration.get_alat();
        const auto& basis = configuration.get_basis();

        auto& neighbour_list = configuration.get_neighbour_list();
        auto& timer = TimerOnce::get();

        const std::size_t max_list_size = configuration.get_max_nl_size();

        neighbour_list.clear();
        neighbour_list.reserve(max_list_size);

        const double r_verlet_cutoff = configuration.get_r_verlet_cutoff();
        const double r_verlet_cutoff_sq = r_verlet_cutoff * r_verlet_cutoff;

        for (auto i{std::size_t{0}}; i < atoms.size() - 1; ++i) {
            const auto& atom_i = atoms[i];

            for (auto j{i + 1}; j < atoms.size(); ++j) {
                const auto& atom_j = atoms[j];
                pair_atoms(configuration, r_verlet_cutoff_sq, atom_i, atom_j, i, j);
            }
        }

        std::cout << neighbour_list.size() << std::endl;

        auto t1{std::chrono::steady_clock::now()};
        timer.update_making_neighbour_list(t1 - t0);
    }

    /**
     * @brief Consider periodic images of a pair and append qualifying pairs to the neighbour list.
     *
     * @param configuration Configuration holding basis/alat and neighbour list storage.
     * @param r_verlet_cutoff_sq Squared Verlet cutoff distance.
     * @param atom_i First atom.
     * @param atom_j Second atom.
     * @param ni Index of atom_i.
     * @param nj Index of atom_j.
     */
    static void pair_atoms(Configuration& configuration, const double r_verlet_cutoff_sq,
                           const Atom& atom_i, const Atom& atom_j, std::size_t ni, std::size_t nj) {
        const double alat = configuration.get_alat();
        const auto& basis = configuration.get_basis();

        auto& neighbour_list = configuration.get_neighbour_list();

        for (auto i{-1}; i <= 1; i++) {
            for (auto j{-1}; j <= 1; j++) {
                for (auto k{-1}; k <= 1; k++) {
                    Maths::Vec3 position_offset{1.0 * i, 1.0 * j, 1.0 * k};

                    const auto r_vec = Maths::Vec3::separation(alat, basis, atom_i.position,
                                                               atom_j.position + position_offset);

                    auto r_sq{r_vec.length_squared()};

                    if (r_sq <= r_verlet_cutoff_sq) {
                        SimpleMD::AtomPair atom_pair{};

                        atom_pair.atom_i_idx = ni;
                        atom_pair.atom_j_idx = nj;

                        atom_pair.r = std::sqrt(r_sq);
                        atom_pair.r = std::max(atom_pair.r, 1.0e-9);

                        auto inv_r{1.0 / atom_pair.r};
                        atom_pair.u_vec = inv_r * r_vec;
                        atom_pair.position_offset = position_offset;

                        neighbour_list.emplace_back(atom_pair);
                    }
                }
            }
        }
    }

    /**
     * @brief Update distances and direction vectors for all neighbour pairs.
     *
     * @param configuration Configuration whose neighbour list entries will be updated.
     */
    static void update_neighbour_list(Configuration& configuration) {
        const auto& atoms = configuration.get_atoms();
        const double alat = configuration.get_alat();
        const auto& basis = configuration.get_basis();

        auto& neighbour_list = configuration.get_neighbour_list();
        auto& timer = TimerOnce::get();

        auto t0{std::chrono::steady_clock::now()};

        for (auto& atom_pair : neighbour_list) {
            const auto& atom_i = atoms[atom_pair.atom_i_idx];
            const auto& atom_j = atoms[atom_pair.atom_j_idx];

            const auto r_vec = Maths::Vec3::separation(alat, basis, atom_i.position,
                                                       atom_j.position + atom_pair.position_offset);

            auto r_sq{r_vec.length_squared()};

            atom_pair.r = std::sqrt(r_sq);

            if (atom_pair.r > 0.0) {
                atom_pair.u_vec = (1.0 / atom_pair.r) * r_vec;
            } else {
                atom_pair.u_vec = {};
            }
        }

        auto t1{std::chrono::steady_clock::now()};
        timer.update_updating_neighbour_list(t1 - t0);
    }

    /**
     * @brief Record the configuration to an XYZ file (ASE/OVITO-compatible format).
     *
     * @param time_step Current time step index.
     * @param xyz_file Output XYZ file path.
     * @param configuration Configuration to record.
     *
     * @throws std::runtime_error If the file cannot be opened for writing.
     */
    static void record_to_xyz(const int time_step, 
                              const Configuration& configuration) {
        const auto& atoms = configuration.get_atoms();
        const double alat = configuration.get_alat();
        const auto& basis = configuration.get_basis();
        
        const auto xyz_file = configuration.get_output_dir() / "out.xyz";

        if (xyz_file.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(xyz_file.parent_path(), ec);
        }

        static bool first_call = true;
        std::ios_base::openmode mode = (first_call || !std::filesystem::exists(xyz_file))
                                           ? (std::ios::out | std::ios::trunc)
                                           : (std::ios::out | std::ios::app);

        std::ofstream out(xyz_file, mode);
        if (!out) {
            throw std::runtime_error("record_to_xyz: failed to open file: " + xyz_file.string());
        }

        out << atoms.size() << '\n';

        out << std::fixed << std::setprecision(6);
        out << "Lattice=\"" << basis[0] * alat << " " << basis[1] * alat << " " << basis[2] * alat
            << "  " << basis[3] * alat << " " << basis[4] * alat << " " << basis[5] * alat << "  "
            << basis[6] * alat << " " << basis[7] * alat << " " << basis[8] * alat << "\" "
            << "Properties=species:S:1:pos:R:3 timestep=" << time_step << " alat=" << alat << '\n';

        out << std::setprecision(10);
        for (const auto& atom : atoms) {
            const auto& f = atom.position;
            const Maths::Vec3 cart = alat * (basis * f);

            out << "X " << cart.x << ' ' << cart.y << ' ' << cart.z << '\n';
        }

        out.flush();
        first_call = false;
    }
};

}  // namespace SimpleMD

#endif
