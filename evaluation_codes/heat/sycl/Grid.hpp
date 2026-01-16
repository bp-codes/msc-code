#ifndef GRID_HPP
#define GRID_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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

    Grid() {}
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

    // Helper to return 1D location given 2D coords (i, j)
    inline std::size_t id(std::size_t i, std::size_t j) const { return j * nx + i; }

    // Helpers to get at, next, thermal diffusivity
    inline double& at(std::size_t i, std::size_t j) { return u[id(i, j)]; }
    inline double at(std::size_t i, std::size_t j) const { return u[id(i, j)]; }
    inline double& nxt(std::size_t i, std::size_t j) { return un[id(i, j)]; }
    inline double a(std::size_t i, std::size_t j) const { return alpha[id(i, j)]; }


    static Grid Load_settings(const nlohmann::json& config_file)
    {
        std::size_t nx = config_file.at("nx").get<std::size_t>();
        std::size_t ny = config_file.at("ny").get<std::size_t>();
        double length_x = config_file.at("length_x").get<double>();
        double length_y = config_file.at("length_y").get<double>();

        Grid model_grid(nx, ny, length_x, length_y);

        return model_grid;
    }


    static void dirichlet_boundaries(Grid& model_grid)
    {
        // Zero Dirichlet boundaries
        for (std::size_t i = 0; i < model_grid.nx; ++i) 
        {
            model_grid.at(i, 0) = 0;
            model_grid.at(i, model_grid.ny - 1) = 0;
        }
        for (std::size_t j = 0; j < model_grid.ny; ++j) 
        {
            model_grid.at(0, j) = 0;
            model_grid.at(model_grid.nx - 1, j) = 0;
        }
    }

    static void grid_to_csv(const std::filesystem::path& fname, const Grid& model_grid, double t)
    {
        std::ofstream f(fname);
        if (!f) 
        {            
            throw std::runtime_error("Cannot open output file: " + fname.string());
        }

        f.setf(std::ios::fixed);
        f << std::setprecision(8);
        f << "# t=" << t << ", nx=" << model_grid.nx << ", ny=" << model_grid.ny
        << ", length_x=" << model_grid.length_x << ", length_y=" << model_grid.length_y << "\n";

        for (std::size_t j = 0; j < model_grid.ny; ++j) 
        {
            for (std::size_t i = 0; i < model_grid.nx; ++i) 
            {
                f << model_grid.at(i, j);
                if (i + 1 < model_grid.nx) f << ",";
            }
            f << "\n";
        }
    }



};





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
    const double base = jalpha["value"].get<double>();
    if (!(base > 0.0)) 
    {
        throw std::runtime_error("alpha.value must be > 0");
    }
    std::fill(model_grid.alpha.begin(), model_grid.alpha.end(), base);

    if (!jalpha.contains("custom")) return;
    const nlohmann::json& customs = jalpha["custom"];
    if (!customs.is_array()) 
    {
        throw std::runtime_error("alpha.custom must be an array");
    }

    for (std::size_t r = 0; r < customs.size(); ++r) 
    {
        const nlohmann::json& rect = customs[r];
        for (const char* k : {"x_min","x_max","y_min","y_max","value"}) 
        {
            if (!rect.contains(k) || !rect[k].is_number()) 
            {
                throw std::runtime_error(std::string("alpha.custom[") + std::to_string(r) +
                                         "] missing numeric field: " + k);
            }
        }

        const double x_min = rect["x_min"].get<double>();
        const double x_max = rect["x_max"].get<double>();
        const double y_min = rect["y_min"].get<double>();
        const double y_max = rect["y_max"].get<double>();
        const double aval  = rect["value"].get<double>();

        if (!(aval > 0.0)) 
        {
            throw std::runtime_error("alpha.custom value must be > 0");
        }

        if (x_max < x_min || y_max < y_min) 
        {
            throw std::runtime_error("alpha.custom has max < min");
        }

        const double xminc = std::max(0.0, std::min(x_min, model_grid.length_x));
        const double xmaxc = std::max(0.0, std::min(x_max, model_grid.length_x));
        const double yminc = std::max(0.0, std::min(y_min, model_grid.length_y));
        const double ymaxc = std::max(0.0, std::min(y_max, model_grid.length_y));
        if (xmaxc < xminc || ymaxc < yminc) continue;

        for (std::size_t j = 0; j < model_grid.ny; ++j) 
        {
            const double y = j * model_grid.dy;
            if (y < yminc || y > ymaxc) continue;
            for (std::size_t i = 0; i < model_grid.nx; ++i) 
            {
                const double x = i * model_grid.dx;
                if (x < xminc || x > xmaxc) continue;
                model_grid.alpha[model_grid.id(i, j)] = aval;
            }
        }
    }
}


#endif