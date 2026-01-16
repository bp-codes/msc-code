#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP


/*********************************************************************************************************************************/
#include <array>
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
#include "Atom.hpp"
#include "Morse.hpp"
/*********************************************************************************************************************************/
namespace SimpleMD
{


class Configuration
{

private:

    std::string _device {};
    double _heat {0.0};
    double _alat {1.0};
    std::array<double, 9> _basis = {
        1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0
    };
    std::size_t _crystal_size {};
    double _r_cutoff {1.0};
    double _r_verlet_cutoff {1.0};
    std::vector<Atom> _atoms {};
    std::vector<AtomPair> _neighbour_list {};
    double _dt {1.0};
    std::size_t _time_steps {};
    std::size_t _rebuild_every {};
    std::size_t _xyz_every {};
    std::size_t _max_nl_size {};
    

public:

    STRING_SET_GET(device);
    DOUBLE_SET_GET(heat);
    DOUBLE_SET_GET(alat);
    ARRAY9_SET_GET(basis);
    SIZE_T_SET_GET(crystal_size);
    DOUBLE_SET_GET(r_cutoff);
    DOUBLE_SET_GET(r_verlet_cutoff);
    CLASS_SET_GET(std::vector<Atom>, atoms);
    CLASS_SET_GET(std::vector<AtomPair>, neighbour_list);
    DOUBLE_SET_GET(dt);
    SIZE_T_SET_GET(time_steps);
    SIZE_T_SET_GET(rebuild_every);
    SIZE_T_SET_GET(xyz_every);
    SIZE_T_SET_GET(max_nl_size);


    inline size_t size() { return _atoms.size(); }


    void display()
    {
        std::cout << "Atoms:            " << _atoms.size() << std::endl;
        std::cout << "Pairs:            " << _neighbour_list.size() << std::endl;
    }

};


// Singleton
SINGLETON(Configuration)

}
#endif