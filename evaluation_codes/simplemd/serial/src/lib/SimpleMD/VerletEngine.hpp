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

        // Zero forces on atoms
        for(auto& atom : atoms)
        {
            atom.force = {0.0, 0.0, 0.0};
        }


        // Calculate forces on atoms
        for(auto& atom_pair : neighbour_list)
        {

            if(atom_pair.r <= r_cutoff)
            {
                // Load atoms and ghost atom
                auto& atom_i = atoms[atom_pair.atom_i_idx];
                auto& atom_j = atoms[atom_pair.atom_j_idx];

                // Calculate force
                const auto force = SimpleMD::Morse::force(atom_pair.r, 0.343, 1.44, 2.863);
                const auto vforce = force * atom_pair.u_vec;

                atom_i.force -= vforce;
                atom_j.force += vforce;
            }

        }
        auto t1 = std::chrono::steady_clock::now();
        timer.update_force_calculations(t1 - t0);

    }

    static void calculate_position(Configuration& configuration)
    {

        auto& atoms = configuration.get_atoms();
        const double dt = configuration.get_dt();

        for(auto& atom : atoms)
        {
            atom.position += atom.velocity * dt + 0.5 * atom.force * atom.inv_mass * dt * dt;      
            atom.position.unit_cell_pbc(); 
        }
    }

    static void calculate_velocity(Configuration& configuration)
    {

        auto& atoms = configuration.get_atoms();
        const double dt = configuration.get_dt();

        // Store starting force in scratch
        for(auto& atom : atoms)
        {
            atom.scratch = atom.force;            
        }

        calculate_force(configuration);

        for(auto& atom : atoms)
        {
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