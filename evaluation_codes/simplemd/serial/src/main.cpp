#ifndef MAIN_CPP
#define MAIN_CPP

/*********************************************************************************************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <filesystem>


#include "lib/SimpleMD/_simplemd.hpp"
#include "lib/Helper/_helper.hpp"
/*********************************************************************************************************************************/


/**
 * @brief Entry point for the SimpleMD executable.
 *
 * Expects a single argument specifying the path to the JSON input configuration file.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 *
 * @return int Exit status code.
 *
 * @throws std::runtime_error Via THROW_RUNTIME_ERROR if input file is not provided.
 */
int main(int argc, char* argv[])
{
    
    // Start timer
    const auto t0 {std::chrono::steady_clock::now()};

    if (argc != 2)
    {
        THROW_RUNTIME_ERROR("must give an input file e.g. ./SimpleMD.x input.json");
    }


    // Run SimpleMD
    const std::filesystem::path input_file {argv[1]};
    SimpleMD::Run::run(input_file);


    // End timer and save
    const auto t1 {std::chrono::steady_clock::now()};
    std::filesystem::create_directory("../results");
    auto& timer = TimerOnce::get();

    // Read input file
    nlohmann::json input{};
    std::ifstream in(input_file);
    if (!in)
        throw std::runtime_error("Failed to open input json");
    in >> input;

    const std::string base_file_name = "../results/serial";
    const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";
    const auto time_total {std::chrono::duration<double>(t1 - t0).count()};
    nlohmann::json output;
    output["type"] = "serial";
    output["time"]["total"] = time_total;
    output["time"]["force_calculations"] = timer.get_force_calculations_seconds();
    output["time"]["making_neighbour_list"] = timer.get_making_neighbour_list_seconds();
    output["time"]["updating_neighbour_list"] = timer.get_updating_neighbour_list_seconds();
    output["time"]["overall_time"] = timer.get_overall_time_seconds();
    output["input"] = input;
    std::ofstream out(json_file);
    if (!out) throw std::runtime_error("Failed to open output JSON file.");
    out << output.dump(4);

    return 0;

}

#endif

