#ifndef GHOST_ATOM_HPP
#define GHOST_ATOM_HPP

/*********************************************************************************************************************************/
#include <cstddef>   // std::size_t
#include "Helper/_helper.hpp"
#include "Maths/_maths.hpp"
/*********************************************************************************************************************************/


namespace SimpleMD
{

/**
 * @brief Represents a ghost atom used for halo/periodic boundary handling.
 *
 * Stores the original atom identifier, a halo flag, and a positional offset
 * relative to the owning real atom.
 */
struct GhostAtom
{
public:
    std::size_t atom_id {};
    bool halo {};
    Maths::Vec3 position_offset {};

    GhostAtom() = default;

    /**
     * @brief Construct a ghost atom with an id, position offset, and halo flag.
     *
     * @param id Atom identifier.
     * @param x X component of the position offset.
     * @param y Y component of the position offset.
     * @param z Z component of the position offset.
     * @param halo Flag indicating whether this atom is part of the halo region.
     */
    GhostAtom(
        const std::size_t id,
        const float x,
        const float y,
        const float z,
        const bool halo) :
        atom_id(id),
        halo(halo),
        position_offset(x, y, z)
    {
    }
};

}   // namespace SimpleMD

#endif
