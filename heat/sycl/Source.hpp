#ifndef SOURCE_HPP
#define SOURCE_HPP

#include <iostream>
#include <vector>
#include <string>
#include "json.hpp"
#include "Grid.hpp"


struct Source 
{
    enum class SpatialKind { Gaussian, Block, Point };
    enum class TemporalKind { Constant, Rate, Impulse };

    SpatialKind spatial_kind{SpatialKind::Gaussian};
    TemporalKind temporal_kind{TemporalKind::Constant};


    double t0{0.0};
    double duration{0.0};
    double amplitude{0.0};

    // Gaussian params
    double x0{0.0};
    double y0{0.0};
    double sigma{0.0};

    // block params
    double x_min{0.0}, x_max{0.0}, y_min{0.0}, y_max{0.0};
};


static std::vector<Source> parse_sources(const nlohmann::json& config_file, const Grid& model_grid)
{
    std::vector<Source> sources;
    if (!config_file.contains("source")) return sources;     // optional
    const nlohmann::json& arr = config_file.at("source");

    if (!arr.is_array()) {
        throw std::runtime_error("\"source\" must be an array");
    }

    for (std::size_t k = 0; k < arr.size(); ++k) 
    {
        const nlohmann::json& js = arr[k];

        if (!js.contains("spatial_kind")) throw std::runtime_error("source item missing \"spatial_kind\"");
        std::string spatial_kind = js.at("spatial_kind").get<std::string>();

        if (!js.contains("temporal_kind")) throw std::runtime_error("source item missing \"temporal_kind\"");
        std::string temporal_kind = js.at("temporal_kind").get<std::string>();

        Source s;
        s.t0       = js.value("t0", 0.0);
        s.duration = js.value("duration", 0.0);
        s.amplitude= js.value("amplitude", 0.0);

        // Spatial kind

        if (spatial_kind == "gaussian") 
        {
            s.spatial_kind = Source::SpatialKind::Gaussian;
            s.x0   = js.value("x0", 0.5*model_grid.length_x);
            s.y0   = js.value("y0", 0.5*model_grid.length_y);
            if (js.contains("sigma")) {
                s.sigma = js.at("sigma").get<double>();
            } else {
                double frac = js.value("sigma_fraction", 0.1);
                s.sigma = frac * std::min(model_grid.length_x, model_grid.length_y);
            }
            if (!(s.sigma > 0.0)) throw std::runtime_error("gaussian source needs positive sigma");
        }
        else if (spatial_kind == "point") 
        {
            s.spatial_kind = Source::SpatialKind::Point;
            s.x0   = js.value("x0", 0.5*model_grid.length_x);
            s.y0   = js.value("y0", 0.5*model_grid.length_y);
        }
        else if (spatial_kind == "block") 
        {
            s.spatial_kind  = Source::SpatialKind::Block;
            if (!(js.contains("x_min") && js.contains("x_max") && js.contains("y_min") && js.contains("y_max")))
            {
                throw std::runtime_error("block source needs x_min/x_max/y_min/y_max");
            }

            s.x_min = js.at("x_min").get<double>();
            s.x_max = js.at("x_max").get<double>();
            s.y_min = js.at("y_min").get<double>();
            s.y_max = js.at("y_max").get<double>();
            if (s.x_max < s.x_min || s.y_max < s.y_min)
                throw std::runtime_error("block has max < min");
            // Clamp to domain
            s.x_min = std::max(0.0, std::min(s.x_min, model_grid.length_x));
            s.x_max = std::max(0.0, std::min(s.x_max, model_grid.length_x));
            s.y_min = std::max(0.0, std::min(s.y_min, model_grid.length_y));
            s.y_max = std::max(0.0, std::min(s.y_max, model_grid.length_y));
        }
        else 
        {
            throw std::runtime_error("unknown source spatial kind: " + spatial_kind);
        }


        // Temporal kind

        if (temporal_kind == "constant") 
        {
            s.temporal_kind = Source::TemporalKind::Constant;
        }
        else if (temporal_kind == "rate") 
        {
            s.temporal_kind = Source::TemporalKind::Rate;
        }
        else 
        {
            throw std::runtime_error("unknown source temporal kind: " + temporal_kind);
        }

        sources.push_back(s);
    }
    return sources;
}


static inline bool is_active(const Source& s, const double t)
{
    // inclusive window [t0, t0+duration]
    return (t >= s.t0) && (t <= s.t0 + s.duration);
}


static inline bool is_not_active(const Source& s, const double t)
{
    // inclusive window [t0, t0+duration]
    return !(t >= s.t0) && (t <= s.t0 + s.duration);
}


static inline double source_value_at(   const Source& s, 
                                        const double t, 
                                        const double x, 
                                        const double y, 
                                        const double dt, 
                                        const double dx, 
                                        const double dy)
{
    if(is_not_active(s, t))
    {
        return 0.0;
    }

    if (s.spatial_kind == Source::SpatialKind::Gaussian) {
        const double r2 = (x - s.x0)*(x - s.x0) + (y - s.y0)*(y - s.y0);
        return s.amplitude * std::exp(-r2 / (2.0*s.sigma*s.sigma));
    }
    else if (s.spatial_kind == Source::SpatialKind::Point)     
    {
        if(x >= s.x0 && x < (s.x0 + dx) && y >= s.y0 && y < (s.y0 + dy))
        {
            return s.amplitude;
        }
    } 
    else 
    { 
        // block
        if (x >= s.x_min && x <= s.x_max && y >= s.y_min && y <= s.y_max) {
            return s.amplitude;
        }
    }
    return 0.0;
}

#endif