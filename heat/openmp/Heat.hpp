#ifndef HEAT_HPP
#define HEAT_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include "json.hpp"
#include "Grid.hpp"
#include "Source.hpp"
#include "Writer.hpp"


class Heat
{

public:


    static void run(std::string& input_file)
    {

        auto start = std::chrono::high_resolution_clock::now();

        Grid model_grid {};
        std::vector<Source> sources {};
        double dt {};
        double t_final {};
        int snapshot_every {};
        std::filesystem::path outdir {};
        std::string prefix {};
        bool queue_output {};

        // Read input
        Heat::read_input(input_file, model_grid, sources, dt, t_final, snapshot_every, outdir, prefix, queue_output);

        // Make output dir
        try 
        {
            std::filesystem::create_directory(outdir);
        } 
        catch (const std::filesystem::filesystem_error& e) 
        {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        // Heat grid
        Heat::heat_grid(model_grid, sources, dt, t_final, snapshot_every, outdir, prefix, queue_output);

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute duration in milliseconds
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Execution time: " << duration.count() << " ms\n";
    }


private:

    // Calculate root squared sum
    static double rss(const std::vector<double>& a) 
    {
        double sum {};
        for (double v : a)
        {
            sum += (v * v) / a.size();
        }
        return std::sqrt(sum);
    }


    static void read_input( std::string& input_file, 
                            Grid& model_grid, 
                            std::vector<Source>& sources , 
                            double& dt, 
                            double& t_final,
                            int& snapshot_every,
                            std::filesystem::path& outdir,
                            std::string& prefix,
                            bool& queue_output
                        )
    {

        // Try reading config file
        std::ifstream in(input_file);
        if (!in) 
        {
            throw std::runtime_error(std::string("Cannot open config file: ") + input_file);
        }

        // Read in
        nlohmann::json config_file;
        in >> config_file;

        // Set up the 2D model grid and load alpha/thermal diffusivity regions
        model_grid = Grid::Load_settings(config_file);
        set_alpha_regions(model_grid, config_file.at("alpha"));

        dt = config_file.value("dt", 0.0);
        t_final = config_file.at("t_final").get<double>();
        if (t_final <= 0.0) throw std::runtime_error("t_final must be > 0");

        snapshot_every = std::max(1, config_file.value("snapshot_every", 100));
        outdir = config_file.value("output_dir", std::filesystem::path("out"));
        prefix = config_file.value("output_prefix", std::string("heat"));
        queue_output = config_file.value("queue_output", true);

        // Zero out the boundaries of the grid
        Grid::dirichlet_boundaries(model_grid);

        // Parse sources (optional)
        sources = parse_sources(config_file, model_grid);

        double alpha_max = *std::max_element(model_grid.alpha.begin(), model_grid.alpha.end());
        if (alpha_max <= 0) throw std::runtime_error("alpha must be > 0");

        double dt_max = 1.0 / (2.0 * alpha_max * (model_grid.invdx2 + model_grid.invdy2));
        if (dt <= 0.0 || dt > dt_max) 
        {
            double chosen = 0.9 * dt_max;
            if (dt > 0.0 && dt > dt_max) 
            {
                std::cerr << "Warning: provided dt=" << dt
                          << " is unstable; using 0.9*dt_max=" << chosen << "\n";
            }
            dt = chosen;
        }
    }


    static void heat_grid(  Grid& model_grid, 
                            const std::vector<Source>& sources , 
                            const double& dt, 
                            const double& t_final,
                            const int& snapshot_every,
                            const std::filesystem::path& outdir,
                            const std::filesystem::path& prefix,
                            const bool queue_output
                        )
    {

        // Start a snapshot writer
        Writer writer;

        // Meta model_grid for snapshot writer
        Grid meta = model_grid;
        meta.u.clear();

        auto t = 0.0;
        auto step = 0;

        // Save grid to file (at t=0)   
        if(queue_output)
        {      
            writer.enqueue(prefix, outdir, t, step, model_grid);
        }
        else
        {
            writer.grid_to_csv(prefix, outdir, t, step, model_grid);
        }


        // Start looping through time steps
        //####################################

        while (t < t_final - 1e-15) 
        {

            // sample sources at midpoint time
            const double t_sample = t + 0.5 * dt;

            // Loop over cells in grid
            #pragma omp parallel for collapse(2) schedule(static)
            for (std::size_t j = 1; j < model_grid.ny - 1; j++) 
            {
                for (std::size_t i = 1; i < model_grid.nx - 1; i++) 
                {
                    // Set x
                    const auto x = i * model_grid.dx;
                    const auto y = j * model_grid.dy;

                    // Value at i, j at time t 
                    const auto uij = model_grid.at(i, j);

                    // 5 point laplacian (second order)     \Delta^2u
                    double lap = (model_grid.at(i+1, j) - 2.0*uij + model_grid.at(i-1, j)) * model_grid.invdx2 +
                                 (model_grid.at(i, j+1) - 2.0*uij + model_grid.at(i, j-1)) * model_grid.invdy2;

                    // Add up contributions from sources S(x,y,t)
                    auto source_accumulator {0.0};
                    auto constant {-1.0};
                    for (const auto& s : sources) 
                    {                        
                        const auto this_source = source_value_at(s, t_sample, x, y, dt, model_grid.dx, model_grid.dy);
                        if(s.temporal_kind == Source::TemporalKind::Rate)
                        {
                            source_accumulator += this_source;
                        }
                        else if(s.temporal_kind == Source::TemporalKind::Constant)
                        {
                            // If there is a constant source, take the maximum
                            constant = std::max(constant, this_source);
                        }
                    }

                    // Get thermal diffusivity at i,j
                    const auto aij = model_grid.a(i, j);

                    // If a constant, just override with this value
                    if(constant > 0.0)
                    {
                        model_grid.nxt(i, j) = constant;
                    }
                    // If not, the value at the next time step is the 
                    // value at this time step + thermal diffusivity * dt * \Delta^2u 
                    //                         + thermal diffusivity * dt * S
                    else
                    {                        
                        model_grid.nxt(i, j) = uij + aij * dt * (lap + source_accumulator);
                    }
                    
                }
            }

            // Enforce dirichlet boundaries
            Grid::dirichlet_boundaries(model_grid);
        
            // Replace u with un before next step
            model_grid.u.swap(model_grid.un);

            // Increment time and step counter
            t += dt;
            step++;

            // Save grid to file (at time t)
            if (step % snapshot_every == 0 || t >= t_final - 1e-15) 
            {           

                if(queue_output)
                {      
                    writer.enqueue(prefix, outdir, t, step, model_grid);
                }
                else
                {
                    writer.grid_to_csv(prefix, outdir, t, step, model_grid);
                }

                // Output message
                std::cerr << "t=" << t << " (step " << step << ")" << std::endl;    

            }
        }
        
        writer.stop();
    }

};


#endif