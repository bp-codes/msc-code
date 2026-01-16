#ifndef SIMPLE_MD_HPP
#define SIMPLE_MD_HPP


/*********************************************************************************************************************************/
#include <json.hpp>
#include "../Helper/_helper.hpp"
#include "../Maths/_maths.hpp"
#include "_simplemd.hpp"
/*********************************************************************************************************************************/
namespace SimpleMD
{



class Run
{

public:

    static void run(const std::filesystem::path& input_file)
    {
        std::cout << "Simple MD" << std::endl;
        
        // Start config
        Configuration& configuration = SimpleMD::ConfigurationOnce::get();
        auto& timer = TimerOnce::get();

        // Load input file
        load_json(input_file, configuration);

        // Display
        configuration.display();
        
        ConfigurationEngine::perturb(configuration.get_crystal_size(), 0.01, configuration);

        // Timed section - time steps in simulation

        auto t0 = std::chrono::steady_clock::now();
        for(int i=0; i<configuration.get_time_steps(); i++)
        {

            if(i%configuration.get_rebuild_every() == 0) ConfigurationEngine::make_neighbour_list(configuration);
            VerletEngine::vertlet_step(configuration);

            if(i%configuration.get_xyz_every() == 0) ConfigurationEngine::record_to_xyz(i, std::filesystem::path("out.xyz"), configuration);

        }
        auto t1 = std::chrono::steady_clock::now();
        
        timer.update_overall_time(t1 - t0);
        timer.print_times();

    }


    void static load_json(const std::filesystem::path& input_file, Configuration& configuration)
    {

        // Load JSON
        if (!std::filesystem::exists(input_file)) 
        {
            THROW_RUNTIME_ERROR("JSON file does not exist: " + input_file.string());
        }

        // Try to open the file
        std::ifstream ifs(input_file);
        if (!ifs.is_open()) 
        {
            THROW_RUNTIME_ERROR("Could not open JSON file for reading: " + input_file.string());
        }

        nlohmann::json config;

        try 
        {
            ifs >> config;      // may throw json::parse_error
        }
        catch (const nlohmann::json::parse_error& e) 
        {
            THROW_RUNTIME_ERROR("Failed to parse JSON file \"" + input_file.string() + "\"");
        }

        // Save variables
        std::size_t threads = Run::load<std::size_t>(config, {"settings", "threads"});
        std::string device = Run::load<std::string>(config, {"settings", "device"});

        double heat = Run::load<double>(config, {"crystal", "heat"});
        std::string crystal_structure = Run::load<std::string>(config, {"crystal", "structure"});
        double alat = Run::load<double>(config, {"crystal", "alat"});
        std::size_t n = Run::load<std::size_t>(config, {"crystal", "size"});
        std::vector<double> ux = Run::load<std::vector<double>>(config, {"crystal", "ux"});
        std::vector<double> uy = Run::load<std::vector<double>>(config, {"crystal", "uy"});
        std::vector<double> uz = Run::load<std::vector<double>>(config, {"crystal", "uz"});

        double r_cutoff = Run::load<double>(config, {"simulation", "r_cutoff"});
        double r_verlet_cutoff = Run::load<double>(config, {"simulation", "r_verlet_cutoff"});
        std::size_t rebuild_every = Run::load<std::size_t>(config, {"simulation", "rebuild_every"});
        double dt = Run::load<double>(config, {"simulation", "dt"});
        std::size_t time_steps = Run::load<std::size_t>(config, {"simulation", "time_steps"});
        std::size_t xyz_every = Run::load<std::size_t>(config, {"simulation", "xyz_every"});
        std::size_t max_nl_size = Run::load<std::size_t>(config, {"simulation", "max_nl_size"});

        // Basis
        std::array<double, 9> basis {
                            ux[0], ux[1], ux[2],
                            uy[0], uy[1], uy[2],
                            uz[0], uz[1], uz[2]
                        };

        // Build atoms
        std::vector<Atom> atoms {};
        if(crystal_structure == "fcc")
        {
            atoms = SimpleMD::Fcc::make("Al", n, n, n);
        }
        else if(crystal_structure == "bcc")
        {
            atoms = SimpleMD::Bcc::make("Al", n, n, n);
        }
        else
        {
            THROW_RUNTIME_ERROR("Crystal structure must be bcc or fcc.");
        }

        // Save to configuration
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



    template<typename T>
    static T load(const nlohmann::json& j, const std::vector<std::string>& keys)
    {
        const nlohmann::json* current = &j;

        for (size_t i = 0; i < keys.size(); ++i) 
        {
            const std::string& key = keys[i];

            if (!current->contains(key)) 
            {
                THROW_RUNTIME_ERROR("Missing key in JSON path: \"" + key + "\"");
            }

            current = &((*current)[key]);  // next
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






}
#endif