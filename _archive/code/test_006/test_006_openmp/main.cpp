#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>



const int nx = 2000;
const int ny = 2000;
const int nt = 1000;        // number of time steps
const double alpha = 1.0;  // thermal diffusivity
const double dx = 1.0;
const double dy = 1.0;
const double dt = 0.1;


// Check stability condition
void checkStability() {
    double coeff = dt * alpha * (1.0 / (dx * dx) + 1.0 / (dy * dy));
    if (coeff >= 0.5) {
        std::cerr << "Warning: Stability condition violated! Reduce dt." << std::endl;
        std::exit(1);
    }
}


// Save the temperature field to a CSV file
void saveToCSV(const std::vector<std::vector<double>>& u, const std::string& filename) {
    std::ofstream file(filename);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            file << u[i][j];
            if (j != ny - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}




int main() 
{


    auto start_setup = std::chrono::high_resolution_clock::now();


    checkStability();

    std::vector<std::vector<double>> u(nx, std::vector<double>(ny, 0.0));
    std::vector<std::vector<double>> u_new(nx, std::vector<double>(ny, 0.0));

    // Initial condition: hot square in the center
    for (int i = nx / 4; i < 3 * nx / 4; ++i) {
        for (int j = ny / 4; j < 3 * ny / 4; ++j) {
            u[i][j] = 100.0;
        }
    }
    saveToCSV(u, "heat_input.csv");


    auto end_setup = std::chrono::high_resolution_clock::now();


    auto start_calc = std::chrono::high_resolution_clock::now();
    //############################################################



    // Time stepping
    for (int n = 0; n < nt; ++n) 
    {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                u_new[i][j] = u[i][j] + alpha * dt * (
                    (u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j]) / (dx * dx) +
                    (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) / (dy * dy)
                );
            }
        }
        u.swap(u_new);  // update
    }


    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();

    saveToCSV(u, "heat_output.csv");
    std::cout << "Simulation complete. Results saved to heat_output.csv" << std::endl;
    
    // Calculate duration
    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << std::endl;

    return 0;
}

