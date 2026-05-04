#ifndef MORSE_HPP
#define MORSE_HPP

/*********************************************************************************************************************************/
#include <cmath>  // std::exp

#include "Helper/_helper.hpp"
#include "SimpleMD/Atom.hpp"

/*********************************************************************************************************************************/

namespace SimpleMD {

/**
 * @brief Provides Morse potential energy and force calculations.
 *
 * The Morse potential is commonly used to model interatomic interactions.
 */
class Morse {
public:
    /**
     * @brief Compute the Morse potential energy.
     *
     * @param r Interatomic separation distance.
     * @param De Dissociation energy.
     * @param a Potential width parameter.
     * @param re Equilibrium separation distance.
     *
     * @return double Potential energy value.
     */
    [[nodiscard]]
    inline static double potential(double r, double De, double a, double re) noexcept {
        const double x = std::exp(-a * (r - re));
        const double one_minus_x = 1.0 - x;
        return De * one_minus_x * one_minus_x;
    }

    /**
     * @brief Compute the Morse force magnitude.
     *
     * This is the negative derivative of the Morse potential with respect to distance.
     *
     * @param r Interatomic separation distance.
     * @param De Dissociation energy.
     * @param a Potential width parameter.
     * @param re Equilibrium separation distance.
     *
     * @return double Force magnitude.
     */
    [[nodiscard]]
    inline static double force(double r, double De, double a, double re) noexcept {
        if (r <= 0.0) {
            return 0.0;
        }

        const double x = std::exp(-a * (r - re));
        return -2.0 * a * De * (1.0 - x) * x;
    }
};

}  // namespace SimpleMD

#endif
