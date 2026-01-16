#ifndef ATOM_PAIR_HPP
#define ATOM_PAIR_HPP

/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
#include "Atom.hpp"
/*********************************************************************************************************************************/

namespace SimpleMD
{

struct AtomPair 
{

public:

    size_t atom_i_idx {};   // Index in atoms
    size_t atom_j_idx {};   // Index in ghost atoms
    double r {};            // Distance between atoms
    Maths::Vec3 u_vec {};   // Unit vector from A to B  
    Maths::Vec3 position_offset {};  // Unit vector from A to B  
        


};

}

#endif 