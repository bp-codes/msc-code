#ifndef MORSE_HPP
#define MORSE_HPP

/*********************************************************************************************************************************/
#include "Helper/_helper.hpp"
#include "Maths/_maths.hpp"
#include "SimpleMD/Atom.hpp"
#include "Helper/CudaHelper.hpp"
#include <cuda_runtime.h>
/*********************************************************************************************************************************/

namespace SimpleMD {

class Morse {
public:
    SIMPLEMD_HOST_DEVICE
    inline static float potential(float r, float De, float a, float re) noexcept {
        const float x = ::exp(-a * (r - re));
        const float one_minus_x = 1.0f - x;
        return De * one_minus_x * one_minus_x;
    }

    SIMPLEMD_HOST_DEVICE
    inline static float force(float r, float De, float a, float re) noexcept {
        if (r <= 0.0f)
            return 0.0f;
        const float x = ::exp(-a * (r - re));
        return -2.0f * a * De * (1.0f - x) * x;
    }
};

}  // namespace SimpleMD

#endif
