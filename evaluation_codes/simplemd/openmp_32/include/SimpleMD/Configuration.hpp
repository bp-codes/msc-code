#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

/*********************************************************************************************************************************/
#include <array>

#include "Helper/_helper.hpp"
#include "Maths/_maths.hpp"
#include "SimpleMD/Atom.hpp"
#include "SimpleMD/Morse.hpp"

/*********************************************************************************************************************************/
namespace SimpleMD {

class Configuration {
private:
    std::string _device{};
    std::filesystem::path _output_dir {};
    float _heat{0.0f};
    float _alat{1.0f};
    std::array<float, 9> _basis = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    std::size_t _crystal_size{};
    float _r_cutoff{1.0f};
    float _r_verlet_cutoff{1.0f};
    std::vector<Atom> _atoms{};
    std::vector<AtomPair> _neighbour_list{};
    float _dt{1.0f};
    std::size_t _time_steps{};
    std::size_t _rebuild_every{};
    std::size_t _xyz_every{};
    std::size_t _max_nl_size{};

public:
    STRING_SET_GET(device);
    CLASS_SET_GET(std::filesystem::path, output_dir);
    FLOAT_SET_GET(heat);
    SIZE_T_SET_GET(crystal_size);
    FLOAT_SET_GET(alat);
    ARRAY9F_SET_GET(basis);
    FLOAT_SET_GET(r_cutoff);
    FLOAT_SET_GET(r_verlet_cutoff);
    CLASS_SET_GET(std::vector<Atom>, atoms);
    CLASS_SET_GET(std::vector<AtomPair>, neighbour_list);
    FLOAT_SET_GET(dt);
    SIZE_T_SET_GET(time_steps);
    SIZE_T_SET_GET(rebuild_every);
    SIZE_T_SET_GET(xyz_every);
    SIZE_T_SET_GET(max_nl_size);

    inline size_t size() {
        return _atoms.size();
    }

    void display() {
        std::cout << "Atoms:            " << _atoms.size() << std::endl;
        std::cout << "Pairs:            " << _neighbour_list.size() << std::endl;
    }
};

// Singleton
SINGLETON(Configuration)

}  // namespace SimpleMD
#endif
