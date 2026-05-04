#ifndef MORSE_HPP
#define MORSE_HPP

/*********************************************************************************************************************************/
#include "Helper/_helper.hpp"
#include "SimpleMD/Atom.hpp"
/*********************************************************************************************************************************/

namespace SimpleMD
{
    
class Morse
{

public:

    inline static float potential(float r, float De, float a, float re) noexcept 
    {
        const float x = std::exp(-a * (r - re));
        const float one_minus_x = 1.0f - x;
        return De * one_minus_x * one_minus_x;
    }

    inline static float force(float r, float De, float a, float re) noexcept 
    {
        if (r <= 0.0f) return 0.0f;
        // F(r) = -dV/dr = -2 a De (1 - e^{-a(r-re)}) e^{-a(r-re)}
        const float x = std::exp(-a * (r - re));
        return -2.0f * a * De * (1.0f - x) * x;
    }

};


}

#endif 