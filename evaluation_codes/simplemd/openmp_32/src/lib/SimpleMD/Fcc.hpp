#ifndef FCC_HPP
#define FCC_HPP


/*********************************************************************************************************************************/
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
#include "Atom.hpp"
/*********************************************************************************************************************************/

namespace SimpleMD
{

class Fcc
{

public:

    static std::vector<Atom> make(const std::string& element, const int nx, const int ny, const int nz)
    {
        std::vector<Atom> result {};
        result.reserve(4 * nx * ny * nz);

        auto x {0.0f};
        auto y {0.0f};
        auto z {0.0f};

        auto n = std::size_t(0);
        auto mass {27.0f};

        for(size_t i = 0; i < nx; i++)
        {
            for(size_t j = 0; j < ny; j++)
            {
                for(size_t k = 0; k < nz; k++)
                {
                    x = (i + 0.25f) / nx;
                    y = (j + 0.25f) / ny;
                    z = (k + 0.25f) / nz;
                    result.emplace_back(Atom {n, x, y, z, mass});
                    ++n;

                    x = (i + 0.25f) / nx;
                    y = (j + 0.75f) / ny;
                    z = (k + 0.75f) / nz;
                    result.emplace_back(Atom {n, x, y, z, mass});
                    ++n;

                    x = (i + 0.75f) / nx;
                    y = (j + 0.25f) / ny;
                    z = (k + 0.75f) / nz;
                    result.emplace_back(Atom {n, x, y, z, mass});
                    ++n;

                    x = (i + 0.75f) / nx;
                    y = (j + 0.75f) / ny;
                    z = (k + 0.25f) / nz;
                    result.emplace_back(Atom {n, x, y, z, mass});
                    ++n;
                }
            }
        }

        return result;
    }


};

}

#endif