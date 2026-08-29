#ifndef SYCLENGINE_HPP
#define SYCLENGINE_HPP

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "heat/Grid.hpp"
#include "heat/Source.hpp"
#include "heat/SyclFunctions.hpp"

#include <nlohmann/json.hpp>
#include <sycl/sycl.hpp>

inline float source_value_at_device(const Source& s, float t, float x, float y, float dt, float dx,
                                    float dy) {
    // Temporal gating
    bool active = false;
    switch (s.temporal_kind) {
        case Source::TemporalKind::Constant:
        case Source::TemporalKind::Rate: {
            active = (t >= s.t0) && (t < s.t0 + s.duration);
            break;
        }
        case Source::TemporalKind::Impulse: {
            // discrete impulse in the step that contains t0
            active = (t >= s.t0) && (t < s.t0 + dt);
            break;
        }
    }
    if (!active)
        return 0.0f;

    // Spatial profile
    float spatial = 0.0f;
    switch (s.spatial_kind) {
        case Source::SpatialKind::Gaussian: {
            const float dx0 = x - s.x0;
            const float dy0 = y - s.y0;
            const float two_sigma2 = 2.0f * s.sigma * s.sigma + 1.0e-38f;
            spatial = Maths::exp(-(dx0 * dx0 + dy0 * dy0) / two_sigma2);
            break;
        }
        case Source::SpatialKind::Block: {
            spatial = (x >= s.x_min && x <= s.x_max && y >= s.y_min && y <= s.y_max) ? 1.0f : 0.0f;
            break;
        }
        case Source::SpatialKind::Point: {
            // Act only on the cell that contains (x0, y0)
            const float hx = 0.5f * dx;
            const float hy = 0.5f * dy;
            spatial = (Maths::fabs(x - s.x0) <= hx && Maths::fabs(y - s.y0) <= hy) ? 1.0f : 0.0f;
            break;
        }
    }
    return s.amplitude * spatial;
}

struct SyclEngine {
    sycl::queue q{};

    std::size_t nx{};
    std::size_t ny{};
    std::size_t num_sources{};

    std::size_t* d_nx{nullptr};
    std::size_t* d_ny{nullptr};
    float* d_dx{nullptr};
    float* d_dy{nullptr};
    float* d_length_x{nullptr};
    float* d_length_y{nullptr};
    float* d_invdx2{nullptr};
    float* d_invdy2{nullptr};

    float* d_u{nullptr};
    float* d_un{nullptr};
    float* d_alpha{nullptr};

    Source* d_sources{nullptr};
    std::size_t* d_num_sources{nullptr};

    // Constructor - save attributes and call allocate
    SyclEngine(const std::string& device, std::size_t nx_in, std::size_t ny_in,
               std::size_t num_sources_in)
        : nx(nx_in), ny(ny_in), num_sources(num_sources_in) {
        // Make queue and allocate memory on device
        _make_queue(device);
        _allocate();
    }

    ~SyclEngine() {
        _cleanup();
    }

    // host -> device
    void upload_grid(const Grid& model_grid, const std::vector<Source>& sources) {
        // Check dimensions ok or throw error
        _check_data_dims(model_grid);
        _check_sources_dims(sources);

        const auto n = model_grid.nx * model_grid.ny;

        *d_nx = model_grid.nx;
        *d_ny = model_grid.ny;
        *d_dx = model_grid.dx;
        *d_dy = model_grid.dy;

        *d_length_x = model_grid.length_x;
        *d_length_y = model_grid.length_y;
        *d_invdx2 = model_grid.invdx2;
        *d_invdy2 = model_grid.invdy2;

        // Copy grids (u, un, thermal diffusivity)
        q.memcpy(d_u, model_grid.u.data(), sizeof(float) * n).wait();
        q.memcpy(d_un, model_grid.un.data(), sizeof(float) * n).wait();
        q.memcpy(d_alpha, model_grid.alpha.data(), sizeof(float) * n).wait();

        // Store number of sources locally and on device
        num_sources = sources.size();
        *d_num_sources = num_sources;
        q.memcpy(d_sources, sources.data(), sizeof(Source) * num_sources).wait();
    }

    // Copy important data from device -> host
    void download_grid(Grid& model_grid) {
        _check_data_dims(model_grid);

        const auto n = model_grid.nx * model_grid.ny;

        q.memcpy(model_grid.u.data(), d_u, sizeof(float) * n).wait();
        q.memcpy(model_grid.un.data(), d_un, sizeof(float) * n).wait();
    }

    // Zero boundaries
    void dirichlet_boundaries() {
        const auto NX = *d_nx;
        const auto NY = *d_ny;
        const auto un = d_un;

        // Top / bottom rows
        q.parallel_for(sycl::range<1>(NX), [=](sycl::id<1> ii) {
            const auto i = ii[0];
            un[0 * NX + i] = 0.0f;         // top
            un[(NY - 1) * NX + i] = 0.0f;  // bottom
        });

        // Left / right columns
        q.parallel_for(sycl::range<1>(NY), [=](sycl::id<1> jj) {
            const auto j = jj[0];
            un[j * NX + 0] = 0.0f;         // left
            un[j * NX + (NX - 1)] = 0.0f;  // right
        });

        q.wait();
    }

    void heat_step(float dt, float t_sample) {
        const size_t NX = *d_nx;
        const size_t NY = *d_ny;
        if (NX < 3 || NY < 3)
            return;  // avoid NY-2/NX-2 underflow

        // cache members with non-shadowing names
        float* u = this->d_u;
        float* un = this->d_un;
        float* a = this->d_alpha;  // may be nullptr → use 1.0 below
        const Source* src = this->d_sources;
        const size_t src_count = (this->d_num_sources ? *this->d_num_sources : 0);

        const float invdx2 = *this->d_invdx2;
        const float invdy2 = *this->d_invdy2;
        const float dx = *this->d_dx;
        const float dy = *this->d_dy;

        q.parallel_for(sycl::range<2>(NY - 2, NX - 2), [=](sycl::id<2> ij) {
             const size_t j = ij[0] + 1;
             const size_t i = ij[1] + 1;
             const size_t idx = j * NX + i;

             const float x = i * dx;
             const float y = j * dy;

             const float uij = u[idx];
             const float lap = (u[idx + 1] - 2.0f * uij + u[idx - 1]) * invdx2 +
                               (u[idx + NX] - 2.0f * uij + u[idx - NX]) * invdy2;

             float source_acc = 0.0f;
             float constant = -1.0f;

             if (src && src_count) {
                 for (size_t k = 0; k < src_count; ++k) {
                     const Source s = src[k];  // local copy is safer with CUDA plugin
                     const float val = source_value_at_device(s, t_sample, x, y, dt, dx, dy);
                     if (s.temporal_kind == Source::TemporalKind::Rate)
                         source_acc += val;
                     else if (s.temporal_kind == Source::TemporalKind::Constant)
                         constant = sycl::fmax(constant, val);
                     else /* Impulse */
                         source_acc += val;
                 }
             }

             const float aij = (a ? a[idx] : 1.0f);
             un[idx] = (constant > 0.0f) ? constant : (uij + aij * dt * (lap + source_acc));
         }).wait();
    }

    // Swap u and un device buffers (cheap pointer swap)
    void swap_buffers() {
        std::swap(d_u, d_un);
    }

    // Size
    std::size_t size() {
        return nx * ny;
    }

private:
    // Make queue
    void _make_queue(const std::string& device) {
        if (device == "cpu") {
            q = sycl::queue{sycl::cpu_selector_v};
        } else if (device == "gpu") {
            q = sycl::queue{sycl::gpu_selector_v};
        } else {
            q = sycl::queue{sycl::default_selector_v};
        }

        std::cerr << "Using device: " << q.get_device().get_info<sycl::info::device::name>()
                  << "\n";
    }

    // Allocate memory on the device
    void _allocate() {
        if (nx == 0 || ny == 0)
            throw std::runtime_error("DeviceGrid: zero dimensions");
        const auto n = nx * ny;

        d_nx = sycl::malloc_shared<std::size_t>(1, q);
        d_ny = sycl::malloc_shared<std::size_t>(1, q);
        d_dx = sycl::malloc_shared<float>(1, q);
        d_dy = sycl::malloc_shared<float>(1, q);
        if (!d_nx || !d_ny || !d_dx || !d_dy)
            throw std::bad_alloc{};

        d_length_x = sycl::malloc_shared<float>(1, q);
        d_length_y = sycl::malloc_shared<float>(1, q);
        d_invdx2 = sycl::malloc_shared<float>(1, q);
        d_invdy2 = sycl::malloc_shared<float>(1, q);
        if (!d_length_x || !d_length_y || !d_invdx2 || !d_invdy2)
            throw std::bad_alloc{};

        d_u = sycl::malloc_device<float>(n, q);
        d_un = sycl::malloc_device<float>(n, q);
        d_alpha = sycl::malloc_device<float>(n, q);

        if (!d_u)
            throw std::bad_alloc{};
        if (!d_un)
            throw std::bad_alloc{};
        if (!d_alpha)
            throw std::bad_alloc{};

        d_num_sources = sycl::malloc_shared<std::size_t>(1, q);
        if (!d_num_sources)
            throw std::bad_alloc{};
        if (num_sources < 1)
            throw std::runtime_error("Must have at least one source.");

        const std::size_t nsrc_alloc = std::max<std::size_t>(1, num_sources);
        d_sources = sycl::malloc_device<Source>(nsrc_alloc, q);
        if (!d_sources)
            throw std::bad_alloc{};
    }

    // Clean up device
    void _cleanup() {
        if (d_u) {
            sycl::free(d_u, q);
            d_u = nullptr;
        }
        if (d_un) {
            sycl::free(d_un, q);
            d_un = nullptr;
        }
        if (d_alpha) {
            sycl::free(d_alpha, q);
            d_alpha = nullptr;
        }
    }

    // Check dimensions
    void _check_data_dims(const Grid& model_grid) const {
        const auto n = nx * ny;
        if (model_grid.nx != nx || model_grid.ny != ny) {
            throw std::runtime_error("DeviceGrid: Grid dims mismatch");
        }
        if (model_grid.u.size() != n || model_grid.un.size() != n || model_grid.alpha.size() != n) {
            throw std::runtime_error("DeviceGrid: Grid vector sizes mismatch");
        }
    }

    void _check_sources_dims(const std::vector<Source>& sources) const {
        if (sources.size() != num_sources) {
            throw std::runtime_error("DeviceGrid: Sources size mismatch");
        }
    }
};

#endif  // SYCLENGINE_HPP
