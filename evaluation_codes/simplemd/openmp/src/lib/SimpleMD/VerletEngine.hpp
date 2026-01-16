#ifndef VERLET_ENGINE_HPP
#define VERLET_ENGINE_HPP


/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
#include "Atom.hpp"
#include "Configuration.hpp"
#include "ConfigurationEngine.hpp"
#include "Morse.hpp"
/*********************************************************************************************************************************/
namespace SimpleMD
{


class VerletEngine
{

public:

    static void calculate_force(Configuration& configuration)
    {
        
        auto t0 = std::chrono::steady_clock::now();
        auto& atoms = configuration.get_atoms();

        auto& neighbour_list = configuration.get_neighbour_list(); 
        const auto r_cutoff = configuration.get_r_cutoff();
        auto& timer = TimerOnce::get();

        const std::size_t n_atoms = atoms.size();
        const std::size_t n_pairs = neighbour_list.size();

        // Zero forces on atoms
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n_atoms); ++i)
        {
            atoms[i].force = {0.0, 0.0, 0.0};
        }


        #pragma omp parallel
        {
            // One private force array per thread
            std::vector<decltype(atoms[0].force)> forces_private(n_atoms);
            for (std::size_t i = 0; i < n_atoms; ++i) 
            {
                forces_private[i] = {0.0, 0.0, 0.0};
            }

            // Each thread processes neighbour list
            #pragma omp for nowait
            for (int k = 0; k < static_cast<int>(n_pairs); ++k)
            {
                const auto& atom_pair = neighbour_list[k];

                if (atom_pair.r <= r_cutoff)
                {
                    const auto force  = SimpleMD::Morse::force(atom_pair.r, 0.343, 1.44, 2.863);
                    const auto vforce = force * atom_pair.u_vec;

                    const auto i = atom_pair.atom_i_idx;
                    const auto j = atom_pair.atom_j_idx;

                    forces_private[i] -= vforce;
                    forces_private[j] += vforce;
                }
            }

            // Reduce thread-local forces
            #pragma omp critical
            {
                for (std::size_t i = 0; i < n_atoms; ++i)
                {
                    atoms[i].force += forces_private[i];
                }
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        timer.update_force_calculations(t1 - t0);

    }

    static void calculate_position(Configuration& configuration)
    {

        auto& atoms = configuration.get_atoms();
        const double dt = configuration.get_dt();
        const std::size_t n_atoms = atoms.size();

        #pragma omp parallel for
        for (std::size_t i = 0; i < n_atoms; ++i)
        {
            auto& atom = atoms[i];
            atom.position += atom.velocity * dt + 0.5 * atom.force * atom.inv_mass * dt * dt;
            atom.position.unit_cell_pbc();
        }
    }

    static void calculate_velocity(Configuration& configuration)
    {

        auto& atoms = configuration.get_atoms();
        const double dt = configuration.get_dt();
        const std::size_t n_atoms = atoms.size();

        // Store starting force in scratch
        #pragma omp parallel for
        for (std::size_t i = 0; i < n_atoms; ++i)
        {
            atoms[i].scratch = atoms[i].force;
        }

        // Recompute forces 
        calculate_force(configuration);

        // Update velocities using old + new forces
        #pragma omp parallel for
        for (std::size_t i = 0; i < n_atoms; ++i)
        {
            auto& atom = atoms[i];
            atom.velocity += 0.5 * (atom.force + atom.scratch) * atom.inv_mass * dt;
        }

    }




    static void vertlet_step(Configuration& configuration)
    {

        calculate_force(configuration);
        calculate_position(configuration);
        ConfigurationEngine::update_neighbour_list(configuration);
        calculate_velocity(configuration);


    }

};

}

#endif