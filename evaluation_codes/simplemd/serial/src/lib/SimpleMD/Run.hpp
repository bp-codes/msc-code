#ifndef SIMPLE_MD_HPP
#define SIMPLE_MD_HPP

/*********************************************************************************************************************************/
#include <array>        // std::array
#include <chrono>       // std::chrono::steady_clock
#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::path
#include <fstream>      // std::ifstream
#include <iostream>     // std::cout, std::endl
#include <string>       // std::string
#include <vector>       // std::vector
#include <json.hpp>
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
#include "_simplemd.hpp"
/*********************************************************************************************************************************/


namespace SimpleMD
{

/**
 * @brief Entry point utilities for running a SimpleMD simulation.
 */
class Run
{
public:
    /**
     * @brief Run a simulation from an input JSON file.
     *
     * Loads configuration from JSON, perturbs atoms, advances time steps, and records output.
     *
     * @param input_file Path to the JSON input file.
     */
    static void run(const std::filesystem::path& input_file)
    {
        std::cout << "Simple MD" << std::endl;

        Configuration& configuration = SimpleMD::ConfigurationOnce::get();
        auto& timer = TimerOnce::get();

        load_json(input_file, configuration);

        configuration.display();

        ConfigurationEngine::perturb(configuration.get_crystal_size(), 0.01, configuration);

        auto t0 {std::chrono::steady_clock::now()};
        for (auto i {std::size_t{0}}; i < configuration.get_time_steps(); i++)
        {
            if (i % configuration.get_rebuild_every() == 0)
            {
                ConfigurationEngine::make_neighbour_list(configuration);
            }

            VerletEngine::vertlet_step(configuration);

            if (i % configuration.get_xyz_every() == 0)
            {
                ConfigurationEngine::record_to_xyz(static_cast<int>(i), std::filesystem::path("out.xyz"), configuration);
            }
        }
        auto t1 {std::chrono::steady_clock::now()};

        timer.update_overall_time(t1 - t0);
        timer.print_times();
    }

    /**
     * @brief Load configuration values from a JSON input file.
     *
     * @param input_file Path to JSON file.
     * @param configuration Configuration instance to populate.
     *
     * @throws std::runtime_error Via THROW_RUNTIME_ERROR on file/parse/type errors.
     */
    static void load_json(const std::filesystem::path& input_file, Configuration& configuration)
    {
        if (!std::filesystem::exists(input_file))
        {
            THROW_RUNTIME_ERROR("JSON file does not exist: " + input_file.string());
        }

        std::ifstream ifs(input_file);
        if (!ifs.is_open())
        {
            THROW_RUNTIME_ERROR("Could not open JSON file for reading: " + input_file.string());
        }

        nlohmann::json config;

        try
        {
            ifs >> config;
        }
        catch (const nlohmann::json::parse_error& e)
        {
            THROW_RUNTIME_ERROR("Failed to parse JSON file \"" + input_file.string() + "\"");
        }

        // NOTE: Variables below are intentionally retained (even if currently unused) to avoid
        // changing behavior/assumptions around JSON schema.
        auto threads {Run::load<std::size_t>(config, {"settings", "threads"})};
        auto device {Run::load<std::string>(config, {"settings", "device"})};

        auto heat {Run::load<double>(config, {"crystal", "heat"})};
        auto crystal_structure {Run::load<std::string>(config, {"crystal", "structure"})};
        auto alat {Run::load<double>(config, {"crystal", "alat"})};
        auto n {Run::load<std::size_t>(config, {"crystal", "size"})};
        auto ux {Run::load<std::vector<double>>(config, {"crystal", "ux"})};
        auto uy {Run::load<std::vector<double>>(config, {"crystal", "uy"})};
        auto uz {Run::load<std::vector<double>>(config, {"crystal", "uz"})};

        auto r_cutoff {Run::load<double>(config, {"simulation", "r_cutoff"})};
        auto r_verlet_cutoff {Run::load<double>(config, {"simulation", "r_verlet_cutoff"})};
        auto rebuild_every {Run::load<std::size_t>(config, {"simulation", "rebuild_every"})};
        auto dt {Run::load<double>(config, {"simulation", "dt"})};
        auto time_steps {Run::load<std::size_t>(config, {"simulation", "time_steps"})};
        auto xyz_every {Run::load<std::size_t>(config, {"simulation", "xyz_every"})};
        auto max_nl_size {Run::load<std::size_t>(config, {"simulation", "max_nl_size"})};

        std::array<double, 9> basis {
            ux[0], ux[1], ux[2],
            uy[0], uy[1], uy[2],
            uz[0], uz[1], uz[2]
        };

        std::vector<Atom> atoms {};
        if (crystal_structure == "fcc")
        {
            atoms = SimpleMD::Fcc::make("Al", static_cast<int>(n), static_cast<int>(n), static_cast<int>(n));
        }
        else if (crystal_structure == "bcc")
        {
            atoms = SimpleMD::Bcc::make("Al", static_cast<int>(n), static_cast<int>(n), static_cast<int>(n));
        }
        else
        {
            THROW_RUNTIME_ERROR("Crystal structure must be bcc or fcc.");
        }

        configuration.set_crystal_size(n);
        configuration.set_alat(n * alat);
        configuration.set_basis(basis);
        configuration.set_atoms(atoms);
        configuration.set_r_cutoff(r_cutoff);
        configuration.set_r_verlet_cutoff(r_verlet_cutoff);
        configuration.set_dt(dt);
        configuration.set_time_steps(time_steps);
        configuration.set_rebuild_every(rebuild_every);
        configuration.set_xyz_every(xyz_every);
        configuration.set_max_nl_size(max_nl_size);
    }

    /**
     * @brief Load a typed value from a nested JSON object using a key path.
     *
     * @tparam T Output type to read via nlohmann::json::get<T>().
     * @param j JSON document/root object.
     * @param keys Sequence of keys describing the path to the target node.
     *
     * @return T Parsed value.
     *
     * @throws std::runtime_error Via THROW_RUNTIME_ERROR if keys are missing or types mismatch.
     */
    template<typename T>
    [[nodiscard]]
    static T load(const nlohmann::json& j, const std::vector<std::string>& keys)
    {
        const nlohmann::json* current = &j;

        for (auto i {std::size_t{0}}; i < keys.size(); ++i)
        {
            const std::string& key = keys[i];

            if (!current->contains(key))
            {
                THROW_RUNTIME_ERROR("Missing key in JSON path: \"" + key + "\"");
            }

            current = &((*current)[key]);
        }

        try
        {
            return current->get<T>();
        }
        catch (const nlohmann::json::type_error& e)
        {
            THROW_RUNTIME_ERROR("Type error at final key: \"" + keys.back() + "\"");
        }
    }
};

}   // namespace SimpleMD

#endif
