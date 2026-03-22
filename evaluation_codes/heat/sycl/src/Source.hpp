#ifndef SOURCE_HPP
#define SOURCE_HPP

#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "json.hpp"
#include "Grid.hpp"

/**
 * @file Source.hpp
 * @brief Heat source definitions, JSON parsing, and source evaluation helpers.
 *
 * Defines a Source descriptor plus helper functions to:
 * - parse a list of sources from a JSON configuration
 * - test whether a source is active at a given time
 * - evaluate the spatial source contribution at (x,y,t)
 *
 * @warning This header provides free functions in the global namespace.
 */

/**
 * @brief Describes a heat source with spatial and temporal behaviour.
 *
 * Spatial kinds:
 * - Gaussian: centered at (x0,y0) with width sigma
 * - Block: rectangular region [x_min,x_max] x [y_min,y_max]
 * - Point: single-cell region starting at (x0,y0) using grid cell size (dx,dy)
 *
 * Temporal kinds:
 * - Constant, Rate, Impulse (Impulse may be unsupported depending on parsing/solver logic).
 */
struct Source
{
    enum class SpatialKind { Gaussian, Block, Point };
    enum class TemporalKind { Constant, Rate, Impulse };

    SpatialKind spatial_kind{SpatialKind::Gaussian};
    TemporalKind temporal_kind{TemporalKind::Constant};

    // Time, duration, amplitude of source
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

/**
 * @brief Parse heat sources from the JSON configuration.
 *
 * Expects an optional "source" array. Each item must contain:
 * - "spatial_kind": "gaussian" | "block" | "point"
 * - "temporal_kind": "constant" | "rate"
 *
 * Optional fields per source:
 * - t0, duration, amplitude
 * - gaussian: x0, y0, sigma OR sigma_fraction
 * - point: x0, y0
 * - block: x_min, x_max, y_min, y_max
 *
 * @param config_file JSON configuration object.
 * @param model_grid Grid metadata used for defaults and clamping.
 * @return Vector of parsed sources (empty if "source" is absent).
 *
 * @throws std::runtime_error If the schema is invalid or values are inconsistent.
 * @throws nlohmann::json::exception If JSON field access fails (missing keys/type mismatch).
 */
[[nodiscard]]
static std::vector<Source> parse_sources(const nlohmann::json& config_file, const Grid& model_grid)
{
    auto sources {std::vector<Source>{}};
    if (!config_file.contains("source")) return sources; // optional

    const nlohmann::json& arr = config_file.at("source");
    if (!arr.is_array())
    {
        throw std::runtime_error("\"source\" must be an array");
    }

    for (auto k {std::size_t(0)}; k < arr.size(); k++)
    {
        const nlohmann::json& js = arr[k];

        if (!js.contains("spatial_kind")) throw std::runtime_error("source item missing \"spatial_kind\"");
        const auto spatial_kind {js.at("spatial_kind").get<std::string>()};

        if (!js.contains("temporal_kind")) throw std::runtime_error("source item missing \"temporal_kind\"");
        const auto temporal_kind {js.at("temporal_kind").get<std::string>()};

        auto s {Source{}};
        s.t0 = js.value("t0", 0.0);
        s.duration = js.value("duration", 0.0);
        s.amplitude = js.value("amplitude", 0.0);

        // Spatial kind
        if (spatial_kind == "gaussian")
        {
            s.spatial_kind = Source::SpatialKind::Gaussian;
            s.x0 = js.value("x0", 0.5 * model_grid.length_x);
            s.y0 = js.value("y0", 0.5 * model_grid.length_y);

            if (js.contains("sigma"))
            {
                s.sigma = js.at("sigma").get<double>();
            }
            else
            {
                const auto frac {js.value("sigma_fraction", 0.1)};
                s.sigma = frac * std::min(model_grid.length_x, model_grid.length_y);
            }

            if (!(s.sigma > 0.0)) throw std::runtime_error("gaussian source needs positive sigma");
        }
        else if (spatial_kind == "point")
        {
            s.spatial_kind = Source::SpatialKind::Point;
            s.x0 = js.value("x0", 0.5 * model_grid.length_x);
            s.y0 = js.value("y0", 0.5 * model_grid.length_y);
        }
        else if (spatial_kind == "block")
        {
            s.spatial_kind = Source::SpatialKind::Block;
            if (!(js.contains("x_min") && js.contains("x_max") && js.contains("y_min") && js.contains("y_max")))
            {
                throw std::runtime_error("block source needs x_min/x_max/y_min/y_max");
            }

            s.x_min = js.at("x_min").get<double>();
            s.x_max = js.at("x_max").get<double>();
            s.y_min = js.at("y_min").get<double>();
            s.y_max = js.at("y_max").get<double>();

            if (s.x_max < s.x_min || s.y_max < s.y_min)
            {
                throw std::runtime_error("block has max < min");
            }

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

/**
 * @brief Test whether a source is active at time t.
 *
 * Uses an inclusive window: [t0, t0 + duration].
 *
 * @param s Source descriptor.
 * @param t Time.
 * @return True if active, false otherwise.
 */
[[nodiscard]]
static inline bool is_active(const Source& s, const double t) noexcept
{
    return (t >= s.t0) && (t <= s.t0 + s.duration);
}

/**
 * @brief Negated activity test.
 *
 * @note This function preserves the original boolean expression as provided.
 *       (It may not be a strict logical negation of is_active().)
 *
 * @param s Source descriptor.
 * @param t Time.
 * @return Boolean result of the stored expression.
 */
[[nodiscard]]
static inline bool is_not_active(const Source& s, const double t) noexcept
{
    // inclusive window [t0, t0+duration]
    return !(t >= s.t0) && (t <= s.t0 + s.duration);
}

/**
 * @brief Evaluate the source contribution at a given time and point.
 *
 * If is_not_active(s,t) returns true, this returns 0. Otherwise, evaluates the
 * spatial kind:
 * - Gaussian: amplitude * exp(-r^2/(2*sigma^2))
 * - Point: amplitude if (x,y) lies inside [x0,x0+dx) x [y0,y0+dy)
 * - Block: amplitude if (x,y) lies inside [x_min,x_max] x [y_min,y_max]
 *
 * @param s Source descriptor.
 * @param t Time at which to evaluate.
 * @param x x-position.
 * @param y y-position.
 * @param dt Time step (currently unused; kept for API compatibility).
 * @param dx Grid cell size in x (used for Point sources).
 * @param dy Grid cell size in y (used for Point sources).
 * @return Source value at (x,y,t).
 */
[[nodiscard]]
static inline double source_value_at(const Source& s,
                                     const double t,
                                     const double x,
                                     const double y,
                                     const double dt,
                                     const double dx,
                                     const double dy)
{
    (void)dt;

    if (is_not_active(s, t))
    {
        return 0.0;
    }

    if (s.spatial_kind == Source::SpatialKind::Gaussian)
    {
        const auto r2 {(x - s.x0) * (x - s.x0) + (y - s.y0) * (y - s.y0)};
        return s.amplitude * std::exp(-r2 / (2.0 * s.sigma * s.sigma));
    }
    else if (s.spatial_kind == Source::SpatialKind::Point)
    {
        if (x >= s.x0 && x < (s.x0 + dx) && y >= s.y0 && y < (s.y0 + dy))
        {
            return s.amplitude;
        }
    }
    else
    {
        // block
        if (x >= s.x_min && x <= s.x_max && y >= s.y_min && y <= s.y_max)
        {
            return s.amplitude;
        }
    }

    return 0.0;
}

#endif
