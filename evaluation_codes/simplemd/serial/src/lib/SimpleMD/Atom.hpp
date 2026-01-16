#ifndef ATOM_HPP
#define ATOM_HPP


/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
/*********************************************************************************************************************************/

namespace SimpleMD
{

inline constexpr double MIN_MASS {1.0e-20};

struct Atom
{

public:

    std::size_t atom_type {};
    std::size_t atom_id {};

    Maths::Vec3 position {};
    Maths::Vec3 force {};
    Maths::Vec3 velocity {};
    double mass {};
    double inv_mass {};
    Maths::Vec3 scratch {};

    Atom() {}
    Atom(std::size_t id, double x, double y, double z, double mass_in) :
        atom_id(id),
        position(x, y, z)
    {
        mass = std::max(mass_in, SimpleMD::MIN_MASS);
        inv_mass = 1.0 / mass;
    }

    void set_position(Maths::Vec3 position_in)
    {
        position = position_in;
        position.unit_cell_pbc();
    }

    

};

}

#endif