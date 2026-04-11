#ifndef GHOST_ATOM_HPP
#define GHOST_ATOM_HPP


/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
/*********************************************************************************************************************************/

namespace SimpleMD
{

struct GhostAtom
{

public:

    std::size_t atom_id {};
    bool halo {};
    Maths::Vec3 position_offset {};

    GhostAtom() {}
    GhostAtom(const std::size_t id, const double x, const double y, const double z, const bool halo) :
        atom_id(id), position_offset(x, y, z), halo(halo) 
        {}



};

}

#endif