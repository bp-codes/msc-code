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

#include <chrono>    // std::chrono::steady_clock

namespace SimpleMD
{

/**
 * @brief Implements a Velocity-Verlet style integration scheme for SimpleMD.
 *
 * Provides force evaluation and integration updates for position and velocity.
 */
class VerletEngine
{
public:
    /**
     * @brief Compute forces on atoms using the current neighbour list.
     *
     * @param configuration Configuration providing atoms, neighbour list and cutoffs.
     */
    static void calculate_force(Configuration& configuration)
    {
        auto t0 {std::chrono::steady_clock::now()};

        auto& atoms = configuration.get_atoms();

        auto& neighbour_list = configuration.get_neighbour_list();
        const float r_cutoff = configuration.get_r_cutoff();
        auto& timer = TimerOnce::get();

        for (auto& atom : atoms)
        {
            atom.force = {0.0f, 0.0f, 0.0f};
        }

        for (auto& atom_pair : neighbour_list)
        {
            if (atom_pair.r <= r_cutoff)
            {
                auto& atom_i = atoms[atom_pair.atom_i_idx];
                auto& atom_j = atoms[atom_pair.atom_j_idx];

                const auto force = SimpleMD::Morse::force(atom_pair.r, 0.343f, 1.44f, 2.863f);
                const auto vforce = force * atom_pair.u_vec;

                atom_i.force -= vforce;
                atom_j.force += vforce;
            }
        }

        auto t1 {std::chrono::steady_clock::now()};
        timer.update_force_calculations(t1 - t0);
    }

    /**
     * @brief Update atom positions using the current velocities and forces.
     *
     * @param configuration Configuration providing atoms and time step.
     */
    static void calculate_position(Configuration& configuration)
    {
        auto& atoms = configuration.get_atoms();
        const float dt = configuration.get_dt();

        for (auto& atom : atoms)
        {
            atom.position += atom.velocity * dt + 0.5f * atom.force * atom.inv_mass * dt * dt;
            atom.position.unit_cell_pbc();
        }
    }

    /**
     * @brief Update atom velocities using forces at the start and end of the step.
     *
     * @param configuration Configuration providing atoms and time step.
     */
    static void calculate_velocity(Configuration& configuration)
    {
        auto& atoms = configuration.get_atoms();
        const float dt = configuration.get_dt();

        for (auto& atom : atoms)
        {
            atom.scratch = atom.force;
        }

        calculate_force(configuration);

        for (auto& atom : atoms)
        {
            atom.velocity += 0.5f * (atom.force + atom.scratch) * atom.inv_mass * dt;
        }
    }

    /**
     * @brief Perform a single Verlet integration step.
     *
     * @param configuration Configuration to advance by one time step.
     */
    static void vertlet_step(Configuration& configuration)
    {
        calculate_force(configuration);
        calculate_position(configuration);
        ConfigurationEngine::update_neighbour_list(configuration);
        calculate_velocity(configuration);
    }
};

}   // namespace SimpleMD

#endif
