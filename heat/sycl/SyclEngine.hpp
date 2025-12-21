#ifndef SYCLENGINE_HPP
#define SYCLENGINE_HPP

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



inline double source_value_at_device(
    const Source& s,
    double t, double x, double y,
    double dt, double dx, double dy)
{
    // Temporal gating
    bool active = false;
    switch (s.temporal_kind) 
    {
        case Source::TemporalKind::Constant:
        case Source::TemporalKind::Rate:
        {
            active = (t >= s.t0) && (t < s.t0 + s.duration);
            break;
        }
        case Source::TemporalKind::Impulse:
        {
            // discrete impulse in the step that contains t0
            active = (t >= s.t0) && (t < s.t0 + dt);
            break;
        }
    }
    if (!active) return 0.0;

    // Spatial profile
    double spatial = 0.0;
    switch (s.spatial_kind) {
        case Source::SpatialKind::Gaussian: 
        {
            const double dx0 = x - s.x0;
            const double dy0 = y - s.y0;
            const double two_sigma2 = 2.0 * s.sigma * s.sigma + 1e-300;
            spatial = sycl::exp(-(dx0*dx0 + dy0*dy0) / two_sigma2);
            break;
        }
        case Source::SpatialKind::Block:
        {
            spatial = (x >= s.x_min && x <= s.x_max &&
                       y >= s.y_min && y <= s.y_max) ? 1.0 : 0.0;
            break;
        }
        case Source::SpatialKind::Point: {
            // Act only on the cell that contains (x0, y0)
            const double hx = 0.5 * dx;
            const double hy = 0.5 * dy;
            spatial = (sycl::fabs(x - s.x0) <= hx && sycl::fabs(y - s.y0) <= hy) ? 1.0 : 0.0;
            break;
        }
    }
    return s.amplitude * spatial;
}



struct SyclEngine 
{
    sycl::queue q{};

    std::size_t nx {};
    std::size_t ny {};
    std::size_t num_sources {};

    std::size_t* d_nx {nullptr};
    std::size_t* d_ny {nullptr};
    double* d_dx {nullptr};
    double* d_dy {nullptr};
    double* d_length_x {nullptr};
    double* d_length_y {nullptr};
    double* d_invdx2 {nullptr};
    double* d_invdy2 {nullptr};

    double* d_u {nullptr};
    double* d_un {nullptr};
    double* d_alpha {nullptr};

    Source* d_sources {nullptr};
    std::size_t* d_num_sources {nullptr};


    // Constructor - save attributes and call allocate
    SyclEngine(const std::string& device, std::size_t nx_in, std::size_t ny_in, std::size_t num_sources_in) : nx(nx_in), ny(ny_in), num_sources(num_sources_in)
    {
        // Make queue and allocate memory on device
        _make_queue(device);
        _allocate();
    }
    ~SyclEngine() { _cleanup(); }






    // host -> device
    void upload_grid(const Grid& model_grid, const std::vector<Source>& sources)  
    { 
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
        q.memcpy(d_u,  model_grid.u.data(),  sizeof(double)*n).wait(); 
        q.memcpy(d_un,  model_grid.un.data(),  sizeof(double)*n).wait(); 
        q.memcpy(d_alpha,  model_grid.alpha.data(),  sizeof(double)*n).wait(); 
    
        // Store number of sources locally and on device
        num_sources = sources.size();
        *d_num_sources = num_sources;
        q.memcpy(d_sources, sources.data(), sizeof(Source) * num_sources).wait();
        

    }
    
    // Copy important data from device -> host
    void download_grid(Grid& model_grid)
    {
        _check_data_dims(model_grid);

        const auto n = model_grid.nx * model_grid.ny;

        q.memcpy(model_grid.u.data(),  d_u,  sizeof(double)*n).wait();
        q.memcpy(model_grid.un.data(), d_un, sizeof(double)*n).wait();
    }

    // Zero boundaries
    void dirichlet_boundaries()
    {
        const auto NX = *d_nx;  
        const auto NY = *d_ny;
        const auto un = d_un; 

        // Top / bottom rows
        q.parallel_for(sycl::range<1>(NX), [=](sycl::id<1> ii){
            const auto i = ii[0];
            un[0*NX + i] = 0.0;                 // top
            un[(NY - 1)*NX + i] = 0.0;          // bottom
        });

        // Left / right columns
        q.parallel_for(sycl::range<1>(NY), [=](sycl::id<1> jj){
            const auto j = jj[0];
            un[j*NX + 0] = 0.0;                 // left
            un[j*NX + (NX-1)]  = 0.0;           // right
        });

        q.wait();
    }

    void heat_step(double dt, double t_sample)
    {
        const size_t NX = *d_nx;
        const size_t NY = *d_ny;
        if (NX < 3 || NY < 3) return;  // avoid NY-2/NX-2 underflow

        // cache members with non-shadowing names
        double*       u   = this->d_u;
        double*       un  = this->d_un;
        double*       a   = this->d_alpha;      // may be nullptr → use 1.0 below
        const Source* src = this->d_sources;
        const size_t  src_count = (this->d_num_sources ? *this->d_num_sources : 0);

        const double invdx2 = *this->d_invdx2;
        const double invdy2 = *this->d_invdy2;
        const double dx      = *this->d_dx;
        const double dy      = *this->d_dy;

        q.parallel_for(sycl::range<2>(NY - 2, NX - 2), [=](sycl::id<2> ij) {
            const size_t j = ij[0] + 1;
            const size_t i = ij[1] + 1;
            const size_t idx = j*NX + i;

            const double x = i * dx;
            const double y = j * dy;

            const double uij = u[idx];
            const double lap = (u[idx + 1] - 2.0*uij + u[idx - 1]) * invdx2
                            + (u[idx + NX] - 2.0*uij + u[idx - NX]) * invdy2;

            double source_acc = 0.0;
            double constant   = -1.0;

            if (src && src_count) {
                for (size_t k = 0; k < src_count; ++k) {
                    const Source s = src[k];  // local copy is safer with CUDA plugin
                    const double val = source_value_at_device(s, t_sample, x, y, dt, dx, dy);
                    if (s.temporal_kind == Source::TemporalKind::Rate)       source_acc += val;
                    else if (s.temporal_kind == Source::TemporalKind::Constant) constant = sycl::fmax(constant, val);
                    else /* Impulse */                                        source_acc += val;
                }
            }

            const double aij = (a ? a[idx] : 1.0);
            un[idx] = (constant > 0.0) ? constant : (uij + aij * dt * (lap + source_acc));
        }).wait();
    }



    // Swap u and un device buffers (cheap pointer swap)
    void swap_buffers() 
    {
        std::swap(d_u, d_un);
    }

    // Size
    std::size_t size() 
    {
        return nx * ny;
    }


private:

    // Make queue
    void _make_queue(const std::string& device)
    {

        if (device == "cpu") 
        {
            q = sycl::queue{sycl::cpu_selector_v};
        } 
        else if (device == "gpu") 
        {
            q = sycl::queue{sycl::gpu_selector_v};
        } 
        else
        {
            q = sycl::queue{sycl::default_selector_v};
        }

        std::cerr << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    }


    // Allocate memory on the device
    void _allocate() 
    {

        if (nx == 0 || ny == 0) throw std::runtime_error("DeviceGrid: zero dimensions");
        const auto n = nx * ny;

        d_nx = sycl::malloc_shared<std::size_t>(1, q);
        d_ny = sycl::malloc_shared<std::size_t>(1, q);
        d_dx = sycl::malloc_shared<double>(1, q);
        d_dy = sycl::malloc_shared<double>(1, q);
        if (!d_nx || !d_ny || !d_dx || !d_dy) throw std::bad_alloc{};

        d_length_x = sycl::malloc_shared<double>(1, q);
        d_length_y = sycl::malloc_shared<double>(1, q);
        d_invdx2 = sycl::malloc_shared<double>(1, q);
        d_invdy2 = sycl::malloc_shared<double>(1, q);
        if (!d_length_x || !d_length_y || !d_invdx2 || !d_invdy2) throw std::bad_alloc{};

        d_u  = sycl::malloc_device<double>(n, q);
        d_un = sycl::malloc_device<double>(n, q);
        d_alpha = sycl::malloc_device<double>(n, q);

        if (!d_u) throw std::bad_alloc{};
        if (!d_un) throw std::bad_alloc{};
        if (!d_alpha) throw std::bad_alloc{};

        d_num_sources = sycl::malloc_shared<std::size_t>(1, q);
        if (!d_num_sources) throw std::bad_alloc{};
        if (num_sources < 1) throw std::runtime_error("Must have at least one source.");
        
        const std::size_t nsrc_alloc = std::max<std::size_t>(1, num_sources);
        d_sources = sycl::malloc_device<Source>(nsrc_alloc, q);
        if (!d_sources) throw std::bad_alloc{};
        
    }


    // Clean up device
    void _cleanup() 
    {
        if (d_u)  { sycl::free(d_u,  q); d_u  = nullptr; }
        if (d_un) { sycl::free(d_un, q); d_un = nullptr; }
        if (d_alpha) { sycl::free(d_alpha, q); d_alpha = nullptr; }
    }

    // Check dimensions
    void _check_data_dims(const Grid& model_grid) const 
    {
        const auto n = nx * ny;
        if (model_grid.nx != nx || model_grid.ny != ny)
        {
            throw std::runtime_error("DeviceGrid: Grid dims mismatch");
        }
        if (model_grid.u.size() != n || model_grid.un.size() != n || model_grid.alpha.size() != n)
        {
            throw std::runtime_error("DeviceGrid: Grid vector sizes mismatch");
        }
    }

    void _check_sources_dims(const std::vector<Source>& sources) const 
    {
        if (sources.size() != num_sources)
        {
            throw std::runtime_error("DeviceGrid: Sources size mismatch");
        }
    }
    

};



#endif