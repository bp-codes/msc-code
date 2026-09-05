#ifndef VERLET_ENGINE_HPP
#define VERLET_ENGINE_HPP

/*********************************************************************************************************************************/
#include "Helper/_helper.hpp"
#include "Maths/_maths.hpp"
#include "SimpleMD/Atom.hpp"
#include "SimpleMD/Configuration.hpp"
#include "SimpleMD/ConfigurationEngine.hpp"
#include "SimpleMD/Morse.hpp"

/*********************************************************************************************************************************/
namespace SimpleMD {

class VerletEngine {
public:

    static void verlet_step(Configuration& configuration);
    static void calculate_force(Configuration& configuration);
    static void calculate_position(Configuration& configuration);
    static void calculate_velocity(Configuration& configuration);
    // static void print_first_atom_device(const char* label, Configuration& configuration);
};

}  // namespace SimpleMD

#endif
