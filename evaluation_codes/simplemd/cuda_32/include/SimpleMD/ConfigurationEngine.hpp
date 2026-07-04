#ifndef CONFIGURATION_ENGINE_HPP
#define CONFIGURATION_ENGINE_HPP

/*********************************************************************************************************************************/

#include <numeric>
#include <random>

#include "Maths/_maths.hpp"
#include "SimpleMD/Atom.hpp"
#include "SimpleMD/Configuration.hpp"
#include "SimpleMD/Morse.hpp"
#include "SimpleMD/Timer.hpp"

/*********************************************************************************************************************************/
namespace SimpleMD {

class ConfigurationEngine {
private:
    static inline void cuda_check(cudaError_t err, const std::string& message);
public:
    /**
     * Heat the configuration to a target temperature for testing.
     */
static inline void heat(Configuration& configuration,
                            double T_target,
                            double kB = 8.617333262145e-5,
                            std::optional<unsigned> seed = std::nullopt)
    {
        static_cast<void>(T_target);
        static_cast<void>(kB);

        auto& atoms = configuration.get_atoms();
        const float sigma = configuration.get_heat();

        if (atoms.empty()) {
            return;
        }

        std::mt19937 rng(seed ? *seed : std::random_device{}());
        auto normal01 = std::normal_distribution<double>(0.0, 1.0);

        for (auto& atom : atoms) {
            Maths::Vec3 d_position = {
                sigma * static_cast<float>(normal01(rng)),
                sigma * static_cast<float>(normal01(rng)),
                sigma * static_cast<float>(normal01(rng))
            };

            atom.set_position(atom.position + d_position);
        }
    }

    static inline void perturb(const std::size_t n, const float sigma,
                               Configuration& configuration) {
        auto& atoms = configuration.get_atoms();

        if (atoms.empty())
            return;

        // RNG setup
        std::mt19937 rng(42);
        auto normal01 = std::normal_distribution<double>(0.0, 1.0);

        // 1) Draw Maxwell–Boltzmann velocities: each component has variance kB*T/m
        for (auto& atom : atoms) {
            const float new_sigma = sigma * (1.0f / n);
            Maths::Vec3 d_position = {new_sigma * static_cast<float>(normal01(rng)), 
                                      new_sigma * static_cast<float>(normal01(rng)),
                                      new_sigma * static_cast<float>(normal01(rng))};

            atom.set_position(atom.position + d_position);
        }
    }

    // CUDA (moved to cu along with kernels)

    static void upload_to_device(Configuration& configuration);
    static void download_from_device(Configuration& configuration);
    static void free_device(Configuration& configuration);
    static void make_neighbour_list(Configuration& configuration);
    static void update_neighbour_list(Configuration& configuration);
    static void record_to_xyz(const int time_step, Configuration& configuration);

};

}  // namespace SimpleMD

#endif
