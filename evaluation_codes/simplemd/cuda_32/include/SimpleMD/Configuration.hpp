#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "Helper/_helper.hpp"
#include "Maths/_maths.hpp"
#include "SimpleMD/Atom.hpp"
#include "SimpleMD/AtomPair.hpp"
#include "SimpleMD/Morse.hpp"

namespace SimpleMD {

class ConfigurationEngine;
class VerletEngine;

class Configuration {
private:
    std::string _device{};
    std::filesystem::path _output_dir{};

    float _heat{0.0f};
    float _alat{1.0f};

    std::array<float, 9> _basis{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };

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

    // Device data
    Atom* _d_atoms{nullptr};
    AtomPair* _d_neighbour_list{nullptr};
    unsigned long long* _d_neighbour_list_size{nullptr};

    friend class ConfigurationEngine;
    friend class VerletEngine;

public:
    CLASS_SET_GET(std::filesystem::path, output_dir);
    STRING_SET_GET(device);
    FLOAT_SET_GET(heat);
    FLOAT_SET_GET(alat);
    ARRAY9F_SET_GET(basis);
    SIZE_T_SET_GET(crystal_size);
    FLOAT_SET_GET(r_cutoff);
    FLOAT_SET_GET(r_verlet_cutoff);
    CLASS_SET_GET(std::vector<Atom>, atoms);
    CLASS_SET_GET(std::vector<AtomPair>, neighbour_list);
    FLOAT_SET_GET(dt);
    SIZE_T_SET_GET(time_steps);
    SIZE_T_SET_GET(rebuild_every);
    SIZE_T_SET_GET(xyz_every);
    SIZE_T_SET_GET(max_nl_size);

    unsigned long long* get_d_neighbour_list_size() {
        return _d_neighbour_list_size;
    }
    

    Configuration() = default;

    Configuration(const Configuration&) = delete;
    Configuration& operator=(const Configuration&) = delete;

    Configuration(Configuration&&) = delete;
    Configuration& operator=(Configuration&&) = delete;

    inline std::size_t size() const
    {
        return _atoms.size();
    }

    // Could move to constructor (but won't, to keep in line with other examples)
    void initialise_cuda()
    {
        int device_count = 0;

        auto err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err));
        }

        if (device_count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }

        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
        }

        cudaDeviceProp device_properties{};
        err = cudaGetDeviceProperties(&device_properties, 0);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(err));
        }

        std::cerr << "Using CUDA device: " << device_properties.name << '\n';
    }

    void display() const
    {
        std::cout << "Atoms: " << _atoms.size() << '\n';
        std::cout << "Pairs: " << _neighbour_list.size() << '\n';
    }

    ~Configuration()
    {
        cudaFree(_d_atoms);
        cudaFree(_d_neighbour_list);
        cudaFree(_d_neighbour_list_size);
    }
};

SINGLETON(Configuration)

}  // namespace SimpleMD

#endif