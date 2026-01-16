#ifndef HEAT_HPP
#define HEAT_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <sycl/sycl.hpp>
#include "json.hpp"
#include "Grid.hpp"
#include "Source.hpp"
#include "Writer.hpp"
#include "SyclEngine.hpp"

/**
 * @file Heat.hpp
 * @brief Heat equation solver driver using a SYCL backend.
 *
 * Reads a JSON configuration, initializes the simulation grid and sources,
 * sets up a SYCL engine (device buffers), advances the solution in time,
 * and writes periodic snapshots.
 *
 * @warning This class performs filesystem I/O and may throw on invalid input.
 */
class Heat
{
public:

    /**
     * @brief Run a heat simulation using the provided input file.
     *
     * Reads the JSON configuration from @p input_file, initializes the model grid,
     * validates stability constraints, creates the output directory, constructs
     * a SYCL engine for the selected device, runs the solver, and prints execution time.
     *
     * @param input_file Path to the JSON configuration file.
     *
     * @throws std::runtime_error If the config file cannot be opened or contains invalid values.
     * @throws nlohmann::json::exception If required JSON fields are missing or have the wrong type.
     * @throws std::filesystem::filesystem_error If creating the output directory fails.
     */
    static void run(std::string& input_file)
    {
        const auto start {std::chrono::high_resolution_clock::now()};

        auto device {std::string{}};
        auto model_grid {Grid{}};
        auto sources {std::vector<Source>{}};
        auto dt {0.0};
        auto t_final {0.0};
        auto snapshot_every {0};
        auto outdir {std::filesystem::path{}};
        auto prefix {std::filesystem::path{}};
        auto queue_output {false};

        // Read input
        Heat::read_input(input_file, device, model_grid, sources, dt, t_final, snapshot_every, outdir, prefix, queue_output);

        // Make output dir
        try
        {
            std::filesystem::create_directory(outdir);
        }
        catch (const std::filesystem::filesystem_error& e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        // Set up sycl engine (will store grid and sources on device)
        SyclEngine sycl_engine {device, model_grid.nx, model_grid.ny, sources.size()};

        // Heat grid
        Heat::heat_grid_sycl(sycl_engine, model_grid, sources, dt, t_final, snapshot_every, outdir, prefix, queue_output);

        // Record end time
        const auto end {std::chrono::high_resolution_clock::now()};

        // Compute duration in milliseconds
        const std::chrono::duration<double, std::milli> duration {end - start};

        std::cout << "Execution time: " << duration.count() << " ms\n";
    }

private:

    /**
     * @brief Root-mean-square (RMS) magnitude of a vector.
     *
     * Computes sqrt( (1/N) * sum_i a[i]^2 ).
     *
     * @param a Input vector.
     * @return RMS value.
     *
     * @warning If @p a is empty, this performs division by zero (undefined behaviour).
     *         Callers must ensure @p a is non-empty.
     */
    [[nodiscard]]
    static double rss(const std::vector<double>& a)
    {
        auto sum {0.0};
        for (const auto v : a)
        {
            sum += (v * v) / a.size();
        }
        return std::sqrt(sum);
    }

    /**
     * @brief Read and validate the JSON input file and initialize simulation state.
     *
     * Loads the selected device string, grid settings, alpha regions, time-stepping parameters,
     * and output settings; enforces Dirichlet boundaries; parses sources; and enforces a stable
     * time step.
     *
     * @param input_file Path to JSON configuration file.
     * @param device Selected device identifier (output).
     * @param model_grid Grid to initialize.
     * @param sources Vector of parsed sources (may be empty).
     * @param dt Time step (may be overridden for stability).
     * @param t_final Final simulation time (> 0).
     * @param snapshot_every Snapshot frequency in steps (>= 1).
     * @param outdir Output directory.
     * @param prefix Output filename prefix.
     * @param queue_output If true, queue snapshot output; otherwise write synchronously.
     *
     * @throws std::runtime_error If the config file cannot be opened or contains invalid values.
     * @throws nlohmann::json::exception If required JSON fields are missing or have the wrong type.
     */
    static void read_input(
        std::string& input_file,
        std::string& device,
        Grid& model_grid,
        std::vector<Source>& sources,
        double& dt,
        double& t_final,
        int& snapshot_every,
        std::filesystem::path& outdir,
        std::filesystem::path& prefix,
        bool& queue_output)
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

        // Read selected device
        device = config_file.at("device").get<std::string>();

        // Set up the 2D model grid and load alpha/thermal diffusivity regions
        model_grid = Grid::Load_settings(config_file);
        set_alpha_regions(model_grid, config_file.at("alpha"));

        dt = config_file.value("dt", 0.0);
        t_final = config_file.at("t_final").get<double>();
        if (t_final <= 0.0) throw std::runtime_error("t_final must be > 0");

        snapshot_every = std::max(1, config_file.value("snapshot_every", 100));
        outdir = config_file.value("output_dir", std::filesystem::path("out"));
        prefix = config_file.value("output_prefix", std::filesystem::path("heat"));
        queue_output = config_file.value("queue_output", true);

        // Zero out the boundaries of the grid
        Grid::dirichlet_boundaries(model_grid);

        // Parse sources (optional)
        sources = parse_sources(config_file, model_grid);

        const auto alpha_max {*std::max_element(model_grid.alpha.begin(), model_grid.alpha.end())};
        if (alpha_max <= 0) throw std::runtime_error("alpha must be > 0");

        const auto dt_max {1.0 / (2.0 * alpha_max * (model_grid.invdx2 + model_grid.invdy2))};
        if (dt <= 0.0 || dt > dt_max)
        {
            const auto chosen {0.9 * dt_max};
            if (dt > 0.0 && dt > dt_max)
            {
                std::cerr << "Warning: provided dt=" << dt
                          << " is unstable; using 0.9*dt_max=" << chosen << "\n";
            }
            dt = chosen;
        }
    }

    /**
     * @brief Advance the heat equation in time using the SYCL backend and write snapshots.
     *
     * Uploads the grid and sources to the device, advances the solution from t=0 to @p t_final
     * with step size @p dt, applies Dirichlet boundaries on-device, and periodically downloads
     * the grid for output.
     *
     * @param sycl_engine SYCL execution engine holding device buffers and queue.
     * @param model_grid Grid state (updated in-place; downloaded from device when output is needed).
     * @param sources Heat sources (may be empty).
     * @param dt Time step size.
     * @param t_final Final simulation time.
     * @param snapshot_every Snapshot frequency in steps.
     * @param outdir Output directory.
     * @param prefix Output prefix.
     * @param queue_output If true, enqueue output; otherwise write synchronously.
     */
    static void heat_grid_sycl(
        SyclEngine& sycl_engine,
        Grid& model_grid,
        const std::vector<Source>& sources,
        const double& dt,
        const double& t_final,
        const int& snapshot_every,
        const std::filesystem::path& outdir,
        const std::filesystem::path& prefix,
        const bool queue_output)
    {
        // Start a snapshot writer
        Writer writer;

        // Meta model_grid for snapshot writer
        auto meta {model_grid};
        meta.u.clear();

        auto t {0.0};
        auto step {0LL};

        // Save grid to file (at t=0)
        if (queue_output)
        {
            writer.enqueue(prefix, outdir, t, step, model_grid);
        }
        else
        {
            writer.grid_to_csv(prefix, outdir, t, step, model_grid);
        }

        // Upload grid + sources (scalars & arrays)
        sycl_engine.upload_grid(model_grid, sources);
        auto& q = sycl_engine.q;
        (void)q;

        // Start looping through time steps
        //####################################

        while (t < t_final - 1e-15)
        {
            // sample sources at midpoint time
            const auto t_sample {t + 0.5 * dt};

            // Calculate grid at next time step
            sycl_engine.heat_step(dt, t_sample);

            // Zero boundaries
            sycl_engine.dirichlet_boundaries();

            // ---- swap device buffers for next step
            sycl_engine.swap_buffers();

            // Increment time and step counter
            t += dt;
            step++;

            // Save grid to file (at time t)
            if (step % snapshot_every == 0 || t >= t_final - 1e-15)
            {
                // Copy u from device
                sycl_engine.download_grid(model_grid);

                if (queue_output)
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
    }
};

#endif
