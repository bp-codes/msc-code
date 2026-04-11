#ifndef ATOM_HPP
#define ATOM_HPP


/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
/*********************************************************************************************************************************/

namespace SimpleMD
{

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
    Atom(const std::size_t id, const double x, const double y, const double z, const double mass) :
        atom_id(id), position(x, y, z), mass(std::max(mass, 1.0e-20)), inv_mass(1.0 / std::max(mass, 1.0e-20))  
        {}

    void set_position(Maths::Vec3 position_in)
    {
        position = position_in;
        position.unit_cell_pbc();
    }

    

};

}

#endif