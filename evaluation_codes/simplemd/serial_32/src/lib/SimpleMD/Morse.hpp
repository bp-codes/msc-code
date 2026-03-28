#ifndef MORSE_HPP
#define MORSE_HPP

/*********************************************************************************************************************************/
#include <cmath>   // std::exp
#include "../Helper/_helper.hpp"
#include "Atom.hpp"
/*********************************************************************************************************************************/


namespace SimpleMD
{
    
/**
 * @brief Provides Morse potential energy and force calculations.
 *
 * The Morse potential is commonly used to model interatomic interactions.
 */
class Morse
{
public:
    /**
     * @brief Compute the Morse potential energy.
     *
     * @param r Interatomic separation distance.
     * @param De Dissociation energy.
     * @param a Potential width parameter.
     * @param re Equilibrium separation distance.
     *
     * @return float Potential energy value.
     */
    [[nodiscard]]
    inline static float potential(float r, float De, float a, float re) noexcept
    {
        const float x = std::exp(-a * (r - re));
        const float one_minus_x = 1.0f - x;
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
     * @return float Force magnitude.
     */
    [[nodiscard]]
    inline static float force(float r, float De, float a, float re) noexcept
    {
        if (r <= 0.0f)
        {
            return 0.0f;
        }

        const float x = std::exp(-a * (r - re));
        return -2.0f * a * De * (1.0f - x) * x;
    }
};

}   // namespace SimpleMD

#endif
