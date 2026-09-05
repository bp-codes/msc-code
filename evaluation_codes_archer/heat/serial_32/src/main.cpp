/**
 * @file main.cpp
 * @brief Entry point for the 2D heat equation solver using a JSON configuration file.
 *
 * Solves:
 *   u_t = alpha * (u_xx + u_yy) on [0,length_x] x [0,length_y]
 * using explicit FTCS with Dirichlet (u=0) boundaries.
 *
 * Reads all inputs from a JSON configuration file and runs the solver.
 *
 * @author Ben Palmer
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026 Ben Palmer
 * SPDX-License-Identifier: MIT
 */

#include <iostream>
#include <string>

#include "heat/Heat.hpp"
#include "heat/helper.hpp"

#include <nlohmann/json.hpp>

/**
 * @brief Program entry point.
 *
 * Expects a single command-line argument: path to the JSON configuration file.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Exit code (0 on success, non-zero on error).
 */
int main(int argc, char** argv) {
    try {
        // Start timer
        const auto t0{std::chrono::steady_clock::now()};

        // Read in path to configuration file
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " config.json\n";
            return 1;
        }

        auto input_file{std::string{argv[1]}};

        // Run heat
        Heat::run(input_file);

        // End timer and save
        const auto t1{std::chrono::steady_clock::now()};
        std::filesystem::create_directory("../results");

        // Read input file
        nlohmann::json input{};
        std::ifstream in(input_file);
        if (!in)
            throw std::runtime_error("Failed to open input json");
        in >> input;

        const std::string base_file_name = "../results/serial_32_heat";
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";
        const auto time_total{std::chrono::duration<double>(t1 - t0).count()};
        nlohmann::json output;
        output["type"] = "serial_32";
        output["time_total"] = time_total;
        output["max_rss_kb"] = helper::max_rss_kb();
        output["input"] = input;
        std::ofstream out(json_file);
        if (!out)
            throw std::runtime_error("Failed to open output JSON file.");
        out << output.dump(4);

        return 0;
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "[JSON parse error] " << e.what() << "\n";
        return 1;
    } catch (const nlohmann::json::type_error& e) {
        std::cerr << "[JSON type error] " << e.what() << "\n";
        return 1;
    } catch (const nlohmann::json::out_of_range& e) {
        std::cerr << "[JSON out-of-range] " << e.what() << "\n";
        return 1;
    } catch (const std::ios_base::failure& e) {
        std::cerr << "[I/O error] " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
