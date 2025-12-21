#ifndef CUDA_ENGINE_HPP
#define CUDA_ENGINE_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "json.hpp"
#include "Grid.hpp"
#include "Source.hpp"

// =======================================================
//  CUDA ENGINE: Device memory manager + operations
// =======================================================

struct CudaEngine
{
public:
    // ---- Host configuration ----
    std::size_t nx{};
    std::size_t ny{};
    std::size_t num_sources{};

    // ---- Device scalars (single values) ----
    std::size_t* d_nx{nullptr};
    std::size_t* d_ny{nullptr};
    std::size_t* d_num_sources{nullptr};
    double* d_dx{nullptr};
    double* d_dy{nullptr};
    double* d_length_x{nullptr};
    double* d_length_y{nullptr};
    double* d_invdx2{nullptr};
    double* d_invdy2{nullptr};

    // ---- Device arrays ----
    double* d_u{nullptr};
    double* d_un{nullptr};
    double* d_alpha{nullptr};
    Source* d_sources{nullptr};

    // =======================================================
    //  Utility macro for CUDA error checking
    // =======================================================
    #define CUDA_CHECK(expr)                                                     \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            throw std::runtime_error(                                           \
                std::string(#expr) + " failed: " + cudaGetErrorString(_err));   \
        }                                                                       \
    } while (0)

    // =======================================================
    //  Constructors / Destructors
    // =======================================================
    CudaEngine(std::size_t nx_in, std::size_t ny_in, std::size_t num_sources_in)
        : nx(nx_in), ny(ny_in), num_sources(num_sources_in)
    {
        _allocate();
    }

    ~CudaEngine() { _cleanup(); }

    // =======================================================
    //  Host ↔ Device Transfers
    // =======================================================
    void upload_grid(const Grid& model_grid, const std::vector<Source>& sources);
    void download_grid(Grid& model_grid);

    // =======================================================
    //  Device Operations
    // =======================================================
    void dirichlet_boundaries(cudaStream_t stream = 0);
    void heat_step(double dt, double t_mid);
    void swap_buffers() noexcept;

private:
    // =======================================================
    //  Internal Memory Management
    // =======================================================
    void _allocate();
    void _cleanup();

    // =======================================================
    //  Validation Helpers
    // =======================================================
    void _check_data_dims(const Grid& model_grid) const;
    void _check_sources_dims(const std::vector<Source>& sources) const;



};

#endif // CUDA_ENGINE_HPP
