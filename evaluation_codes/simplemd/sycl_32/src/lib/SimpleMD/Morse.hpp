#ifndef MORSE_HPP
#define MORSE_HPP

/*********************************************************************************************************************************/
#include <sycl/sycl.hpp>

#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
#include "Atom.hpp"

/*********************************************************************************************************************************/



namespace SimpleMD
{
    
class Morse
{

public:

    inline static float potential(float r, float De, float a, float re) noexcept 
    {
        const float x = Maths::sycl_compatible_exp(-a * (r - re));
        const float one_minus_x = 1.0 - x;
        return De * one_minus_x * one_minus_x;
    }

    inline static float force(float r, float De, float a, float re) noexcept 
    {
        if (r <= 0.0) return 0.0;
        // F(r) = -dV/dr = -2 a De (1 - e^{-a(r-re)}) e^{-a(r-re)}
        const float x = Maths::sycl_compatible_exp(-a * (r - re));
        return -2.0 * a * De * (1.0 - x) * x;
    }

};


}

#endif 