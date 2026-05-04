#ifndef ATOM_HPP
#define ATOM_HPP

/*********************************************************************************************************************************/
#include <algorithm>   // std::max
#include <cstddef>     // std::size_t
#include "Helper/_helper.hpp"
#include "Maths/_maths.hpp"
/*********************************************************************************************************************************/


namespace SimpleMD
{

inline constexpr float MIN_MASS {1.0e-20f};

/**
 * @brief Represents a particle/atom with state used by the SimpleMD code.
 *
 * Stores identifiers, kinematic state, mass properties, and scratch storage.
 */
struct Atom
{
public:
    std::size_t atom_type {};
    std::size_t atom_id {};

    Maths::Vec3 position {};
    Maths::Vec3 force {};
    Maths::Vec3 velocity {};
    float mass {};
    float inv_mass {};
    Maths::Vec3 scratch {};

    Atom() = default;

    /**
     * @brief Construct an atom with an id, position and mass.
     *
     * The mass is clamped to a minimum value to avoid division by zero.
     *
     * @param id Atom identifier.
     * @param x X position.
     * @param y Y position.
     * @param z Z position.
     * @param mass_in Input mass value (will be clamped to MIN_MASS).
     */
    Atom(std::size_t id, float x, float y, float z, float mass_in) :
        atom_id(id),
        position(x, y, z)
    {
        mass = std::max(mass_in, SimpleMD::MIN_MASS);
        inv_mass = 1.0f / mass;
    }

    /**
     * @brief Set the atom position and apply unit-cell periodic boundary conditions.
     *
     * @param position_in New position.
     */
    void set_position(const Maths::Vec3& position_in)
    {
        position = position_in;
        position.unit_cell_pbc();
    }
};

}   // namespace SimpleMD

#endif
