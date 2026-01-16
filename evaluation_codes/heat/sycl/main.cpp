// heat_solver_json.cpp
// 2-D Heat equation: u_t = alpha * (u_xx + u_yy) on [0,length_x]x[0,length_y]
// Explicit FTCS with Dirichlet (u=0) boundaries.
// Reads all inputs from a JSON configuration file.
//
// Build (single header):
//   g++ -O3 -std=c++17 heat_solver_json.cpp -o heat
//
// Build (system nlohmann-json package):
//   sudo apt-get install nlohmann-json3-dev
//   g++ -O3 -std=c++17 heat_solver_json.cpp -o heat -I/usr/include

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include "json.hpp"
#include "Heat.hpp"




int main(int argc, char** argv)
{
    try 
    {
        
        // Read in path to configuration file     
        if (argc != 2) 
        {
            std::cerr << "Usage: " << argv[0] << " config.json\n";
            return 1;
        }
        std::string input_file = argv[1];
        
        // Run heat
        Heat::run(input_file);

        return 0;
    }
    catch (const nlohmann::json::parse_error& e) 
    {
        std::cerr << "[JSON parse error] " << e.what() << "\n";
        return 1;
    }
    catch (const nlohmann::json::type_error& e) 
    {
        std::cerr << "[JSON type error] " << e.what() << "\n";
        return 1;
    }
    catch (const nlohmann::json::out_of_range& e) 
    {
        std::cerr << "[JSON out-of-range] " << e.what() << "\n";
        return 1;
    }
    catch (const std::ios_base::failure& e) 
    {
        std::cerr << "[I/O error] " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception& e) 
    {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
