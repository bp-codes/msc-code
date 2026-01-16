#ifndef MORSE_HPP
#define MORSE_HPP

/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
#include "Atom.hpp"
/*********************************************************************************************************************************/

namespace SimpleMD
{
    
class Morse
{

public:

    inline static double potential(double r, double De, double a, double re) noexcept 
    {
        const double x = std::exp(-a * (r - re));
        const double one_minus_x = 1.0 - x;
        return De * one_minus_x * one_minus_x;
    }

    inline static double force(double r, double De, double a, double re) noexcept 
    {
        if (r <= 0.0) return 0.0;
        // F(r) = -dV/dr = -2 a De (1 - e^{-a(r-re)}) e^{-a(r-re)}
        const double x = std::exp(-a * (r - re));
        return -2.0 * a * De * (1.0 - x) * x;
    }

};


}

#endif 