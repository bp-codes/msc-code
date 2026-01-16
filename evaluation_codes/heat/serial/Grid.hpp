#ifndef GRID_HPP
#define GRID_HPP

#include <cstddef>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <iomanip>

#include "json.hpp"

/**
 * @file Grid.hpp
 * @brief 2D grid storage and utilities for the heat solver.
 *
 * Provides a lightweight grid with spacing metadata and storage for:
 * - thermal diffusivity (alpha)
 * - solution at current time (u)
 * - solution at next time step (un)
 *
 * Also provides helpers for indexing, boundary conditions, and CSV output.
 */
struct Grid
{
    std::size_t nx{};
    std::size_t ny{};
    double length_x{};
    double length_y{};
    double dx{};
    double dy{};
    double invdx2 {};
    double invdy2 {};
    std::vector<double> alpha;
    std::vector<double> u;
    std::vector<double> un;

    /**
     * @brief Default constructor.
     *
     * Leaves dimensions and arrays empty/zero-initialized.
     */
    Grid() {}

    /**
     * @brief Construct a grid with dimensions and physical extents.
     *
     * Allocates alpha/u/un arrays of size nx*ny and computes spacing metadata.
     *
     * @param nx_in Number of grid points in x (must be >= 3).
     * @param ny_in Number of grid points in y (must be >= 3).
     * @param lx Domain length in x.
     * @param ly Domain length in y.
     *
     * @throws std::runtime_error If nx_in or ny_in are < 3.
     */
    Grid(std::size_t nx_in, std::size_t ny_in, double lx, double ly)
        : nx(nx_in), ny(ny_in), length_x(lx), length_y(ly)
    {
        if (nx < 3 || ny < 3)
        {
            throw std::runtime_error("nx and ny must be >= 3");
        }

        dx = length_x / (nx - 1);
        dy = length_y / (ny - 1);

        invdx2 = 1.0 / (dx * dx);
        invdy2 = 1.0 / (dy * dy);

        alpha.assign(nx * ny, 0.0);
        u.assign(nx * ny, 0.0);
        un.assign(nx * ny, 0.0);
    }

    /**
     * @brief Map 2D indices (i, j) to the 1D storage index.
     *
     * Storage order is row-major: index = j*nx + i.
     *
     * @param i x-index.
     * @param j y-index.
     * @return 1D linear index into alpha/u/un.
     */
    [[nodiscard]]
    inline std::size_t id(std::size_t i, std::size_t j) const noexcept
    {
        return j * nx + i;
    }

    /**
     * @brief Access the solution value u at (i, j).
     *
     * @param i x-index.
     * @param j y-index.
     * @return Reference to u(i, j).
     *
     * @warning No bounds checking is performed.
     */
    inline double& at(std::size_t i, std::size_t j)
    {
        return u[id(i, j)];
    }

    /**
     * @brief Access the solution value u at (i, j) (const overload).
     *
     * @param i x-index.
     * @param j y-index.
     * @return Value of u(i, j).
     *
     * @warning No bounds checking is performed.
     */
    [[nodiscard]]
    inline double at(std::size_t i, std::size_t j) const
    {
        return u[id(i, j)];
    }

    /**
     * @brief Access the next-step solution value un at (i, j).
     *
     * @param i x-index.
     * @param j y-index.
     * @return Reference to un(i, j).
     *
     * @warning No bounds checking is performed.
     */
    inline double& nxt(std::size_t i, std::size_t j)
    {
        return un[id(i, j)];
    }

    /**
     * @brief Access the thermal diffusivity alpha at (i, j).
     *
     * @param i x-index.
     * @param j y-index.
     * @return Value of alpha(i, j).
     *
     * @warning No bounds checking is performed.
     */
    [[nodiscard]]
    inline double a(std::size_t i, std::size_t j) const
    {
        return alpha[id(i, j)];
    }

    /**
     * @brief Construct a Grid from a JSON configuration.
     *
     * Expects fields: "nx", "ny", "length_x", "length_y".
     *
     * @param config_file JSON configuration object.
     * @return Initialized Grid instance.
     *
     * @throws nlohmann::json::exception If required fields are missing or have wrong type.
     * @throws std::runtime_error If nx/ny are invalid (< 3).
     */
    [[nodiscard]]
    static Grid Load_settings(const nlohmann::json& config_file)
    {
        const auto nx {config_file.at("nx").get<std::size_t>()};
        const auto ny {config_file.at("ny").get<std::size_t>()};
        const auto length_x {config_file.at("length_x").get<double>()};
        const auto length_y {config_file.at("length_y").get<double>()};

        auto model_grid {Grid(nx, ny, length_x, length_y)};
        return model_grid;
    }

    /**
     * @brief Apply zero Dirichlet boundary conditions to the solution field.
     *
     * @param model_grid Grid whose solution field will be modified.
     */
    static void dirichlet_boundaries(Grid& model_grid)
    {
        for (auto i {std::size_t(0)}; i < model_grid.nx; i++)
        {
            model_grid.at(i, 0) = 0;
            model_grid.at(i, model_grid.ny - 1) = 0;
        }
        for (auto j {std::size_t(0)}; j < model_grid.ny; j++)
        {
            model_grid.at(0, j) = 0;
            model_grid.at(model_grid.nx - 1, j) = 0;
        }
    }

    /**
     * @brief Write the grid solution field to a CSV file.
     *
     * @param fname Output file path.
     * @param model_grid Grid to write.
     * @param t Simulation time.
     *
     * @throws std::runtime_error If the output file cannot be opened.
     */
    static void grid_to_csv(const std::filesystem::path& fname,
                            const Grid& model_grid,
                            double t)
    {
        std::ofstream f(fname);
        if (!f)
        {
            throw std::runtime_error("Cannot open output file: " + fname.string());
        }

        f.setf(std::ios::fixed);
        f << std::setprecision(8);
        f << "# t=" << t << ", nx=" << model_grid.nx << ", ny=" << model_grid.ny
          << ", length_x=" << model_grid.length_x
          << ", length_y=" << model_grid.length_y << "\n";

        for (auto j {std::size_t(0)}; j < model_grid.ny; j++)
        {
            for (auto i {std::size_t(0)}; i < model_grid.nx; i++)
            {
                f << model_grid.at(i, j);
                if (i + 1 < model_grid.nx) f << ",";
            }
            f << "\n";
        }
    }
};

/**
 * @brief Apply thermal diffusivity (alpha) regions from JSON configuration.
 *
 * @param model_grid Grid whose alpha field will be modified.
 * @param jalpha JSON object describing alpha configuration.
 *
 * @throws std::runtime_error If required fields are missing or invalid.
 * @throws nlohmann::json::exception If JSON access fails.
 */
static void set_alpha_regions(Grid& model_grid, const nlohmann::json& jalpha)
{
    if (!jalpha.is_object())
    {
        throw std::runtime_error("alpha must be an object with at least {\"value\": ...}");
    }

    if (!jalpha.contains("value") || !jalpha["value"].is_number())
    {
        throw std::runtime_error("alpha.value missing or not a number");
    }

    const auto base {jalpha["value"].get<double>()};
    if (!(base > 0.0))
    {
        throw std::runtime_error("alpha.value must be > 0");
    }

    std::fill(model_grid.alpha.begin(), model_grid.alpha.end(), base);

    if (!jalpha.contains("custom")) return;
    const auto& customs {jalpha["custom"]};
    if (!customs.is_array())
    {
        throw std::runtime_error("alpha.custom must be an array");
    }

    for (auto r {std::size_t(0)}; r < customs.size(); r++)
    {
        const auto& rect {customs[r]};
        for (const char* k : {"x_min","x_max","y_min","y_max","value"})
        {
            if (!rect.contains(k) || !rect[k].is_number())
            {
                throw std::runtime_error(
                    std::string("alpha.custom[") + std::to_string(r) +
                    "] missing numeric field: " + k);
            }
        }

        const auto x_min {rect["x_min"].get<double>()};
        const auto x_max {rect["x_max"].get<double>()};
        const auto y_min {rect["y_min"].get<double>()};
        const auto y_max {rect["y_max"].get<double>()};
        const auto aval  {rect["value"].get<double>()};

        if (!(aval > 0.0))
        {
            throw std::runtime_error("alpha.custom value must be > 0");
        }

        if (x_max < x_min || y_max < y_min)
        {
            throw std::runtime_error("alpha.custom has max < min");
        }

        const auto xminc {std::max(0.0, std::min(x_min, model_grid.length_x))};
        const auto xmaxc {std::max(0.0, std::min(x_max, model_grid.length_x))};
        const auto yminc {std::max(0.0, std::min(y_min, model_grid.length_y))};
        const auto ymaxc {std::max(0.0, std::min(y_max, model_grid.length_y))};
        if (xmaxc < xminc || ymaxc < yminc) continue;

        for (auto j {std::size_t(0)}; j < model_grid.ny; j++)
        {
            const auto y {j * model_grid.dy};
            if (y < yminc || y > ymaxc) continue;

            for (auto i {std::size_t(0)}; i < model_grid.nx; i++)
            {
                const auto x {i * model_grid.dx};
                if (x < xminc || x > xmaxc) continue;

                model_grid.alpha[model_grid.id(i, j)] = aval;
            }
        }
    }
}

#endif
