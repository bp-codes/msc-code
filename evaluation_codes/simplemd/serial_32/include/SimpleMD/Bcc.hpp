#ifndef BCC_HPP
#define BCC_HPP

/*********************************************************************************************************************************/
#include <cstddef>     // std::size_t
#include <string>      // std::string
#include <vector>      // std::vector
#include "Helper/_helper.hpp"
#include "Maths/_maths.hpp"
#include "SimpleMD/Atom.hpp"
/*********************************************************************************************************************************/


namespace SimpleMD
{

/**
 * @brief Utility class to generate body-centred cubic (BCC) atom configurations.
 */
class Bcc
{
public:

    /**
     * @brief Generate a BCC lattice of atoms.
     *
     * Creates a normalized BCC lattice with two atoms per unit cell. The lattice is
     * constructed within the unit cube [0,1]^3 using fractional coordinates.
     *
     * @param element Element identifier (currently unused).
     * @param nx Number of unit cells in the x direction.
     * @param ny Number of unit cells in the y direction.
     * @param nz Number of unit cells in the z direction.
     *
     * @return std::vector<Atom> Vector containing generated atoms.
     */
    [[nodiscard]]
    static std::vector<Atom> make(const std::string& element, const int nx, const int ny, const int nz)
    {
        std::vector<Atom> result {};
        result.reserve(2 * nx * ny * nz);

        auto x {0.0f};
        auto y {0.0f};
        auto z {0.0f};

        auto n {std::size_t{0}};
        auto mass {27.0f};

        for (auto i {std::size_t{0}}; i < static_cast<std::size_t>(nx); i++)
        {
            for (auto j {std::size_t{0}}; j < static_cast<std::size_t>(ny); j++)
            {
                for (auto k {std::size_t{0}}; k < static_cast<std::size_t>(nz); k++)
                {
                    x = (i + 0.25f) / nx;
                    y = (j + 0.25f) / ny;
                    z = (k + 0.25f) / nz;
                    result.emplace_back(Atom {n, x, y, z, mass});
                    ++n;

                    x = (i + 0.75f) / nx;
                    y = (j + 0.75f) / ny;
                    z = (k + 0.75f) / nz;
                    result.emplace_back(Atom {n, x, y, z, mass});
                    ++n;
                }
            }
        }

        return result;
    }
};

}   // namespace SimpleMD

#endif
