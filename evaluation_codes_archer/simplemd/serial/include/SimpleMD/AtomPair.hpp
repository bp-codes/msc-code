#ifndef ATOM_PAIR_HPP
#define ATOM_PAIR_HPP

/*********************************************************************************************************************************/
#include <cstddef>  // std::size_t

#include "Helper/_helper.hpp"
#include "Maths/_maths.hpp"
#include "SimpleMD/Atom.hpp"

/*********************************************************************************************************************************/

namespace SimpleMD {

/**
 * @brief Represents a pair of atoms and their geometric relationship.
 *
 * Stores indices of the atom pair along with separation distance, unit direction vector,
 * and position offset. Typically used in neighbour lists and force calculations.
 */
struct AtomPair {
public:
    std::size_t atom_i_idx{};  ///< Index of atom i in the main atom container.
    std::size_t atom_j_idx{};  ///< Index of atom j in the ghost atom container.

    double r{};  ///< Distance between the atoms.

    Maths::Vec3 u_vec{};            ///< Unit vector from atom i to atom j.
    Maths::Vec3 position_offset{};  ///< Position offset vector from atom i to atom j.
};

}  // namespace SimpleMD

#endif
