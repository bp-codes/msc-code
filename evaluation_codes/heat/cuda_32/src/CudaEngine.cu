/**
 * @file CudaEngine.cu
 * @brief CUDA backend implementation for explicit FTCS heat stepping and boundary enforcement.
 *
 * Implements CUDA kernels and CudaEngine methods used by the heat solver:
 * - one explicit interior time step (FTCS with 5-point Laplacian)
 * - Dirichlet boundary enforcement on device
 * - host<->device upload/download of Grid state and Source list
 */

#include "CudaEngine.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>

#include <cuda_runtime.h>

// -------------------- Kernels --------------------

/**
 * @brief Set the top and bottom Dirichlet boundaries (u=0) on the device.
 *
 * @param un Output buffer (next step).
 * @param NX Grid size in x.
 * @param NY Grid size in y.
 */
__global__ void set_top_bottom(float* __restrict__ un,
                               std::size_t NX, std::size_t NY)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX)
    {
        // top row j=0
        un[i] = 0.0f;

        // bottom row j=NY-1
        if (NY > 0) un[(NY - 1) * NX + i] = 0.0f;
    }
}

/**
 * @brief Set the left and right Dirichlet boundaries (u=0) on the device.
 *
 * @param un Output buffer (next step).
 * @param NX Grid size in x.
 * @param NY Grid size in y.
 */
__global__ void set_left_right(float* __restrict__ un,
                               std::size_t NX, std::size_t NY)
{
    const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < NY)
    {
        // left i=0
        un[j * NX] = 0.0f;

        // right i=NX-1
        if (NX > 0) un[j * NX + (NX - 1)] = 0.0f;
    }
}

/**
 * @brief Device-side evaluation of source contribution S(x,y,t).
 *
 * Behaviour matches the host-side interpretation:
 * - active window: [t0, t0+duration)
 * - spatial shapes: Gaussian, Point (single cell), Block
 *
 * @param s Source descriptor.
 * @param t Time at which to evaluate.
 * @param x x-position.
 * @param y y-position.
 * @param dt Time step (currently unused; kept for API compatibility).
 * @param dx Cell size in x (used for Point sources).
 * @param dy Cell size in y (used for Point sources).
 * @return Source value at (x,y,t).
 */
__device__ inline float device_source_value_at(
    const Source& s,
    float t,
    float x,
    float y,
    float dt,
    float dx,
    float dy)
{
    (void)dt;

    // Check active time window
    if (!(t >= s.t0 && t < (s.t0 + s.duration)))
    {
        return 0.0f;
    }

    float val = 0.0f;

    if (s.spatial_kind == Source::SpatialKind::Gaussian)
    {
        const float dx0 = x - s.x0;
        const float dy0 = y - s.y0;
        const float r2  = dx0 * dx0 + dy0 * dy0;
        val = s.amplitude * expf(-r2 / (2.0f * s.sigma * s.sigma));
    }
    else if (s.spatial_kind == Source::SpatialKind::Point)
    {
        // Treat the point as one grid cell wide
        if (x >= s.x0 && x < (s.x0 + dx) &&
            y >= s.y0 && y < (s.y0 + dy))
        {
            val = s.amplitude;
        }
    }
    else // Block
    {
        if (x >= s.x_min && x <= s.x_max &&
            y >= s.y_min && y <= s.y_max)
        {
            val = s.amplitude;
        }
    }

    return val;
}

/**
 * @brief Compute one explicit FTCS heat step on interior cells.
 *
 * Updates un(i,j) for 1<=i<nx-1, 1<=j<ny-1:
 *   un = u + alpha*dt*(laplacian(u) + source)
 * If a Constant temporal source is present, the per-cell update may be overridden
 * by the maximum constant source value at that cell.
 */
__global__ void heat_step_kernel(
    const float* __restrict__ u,      // in
    float* __restrict__ un,           // out
    const float* __restrict__ alpha,  // per-cell diffusivity
    std::size_t nx, std::size_t ny,
    float invdx2, float invdy2,
    float dx, float dy, float dt,
    const Source* __restrict__ sources,
    std::size_t num_sources,
    float t_mid // midpoint time (t + 0.5*dt)
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i <= 0 || j <= 0 || i >= static_cast<int>(nx) - 1 || j >= static_cast<int>(ny) - 1) return;

    const std::size_t idx = static_cast<std::size_t>(j) * nx + static_cast<std::size_t>(i);

    const float uij = u[idx];

    // 5-point Laplacian
    const float lap =
        (u[idx + 1] - 2.0f * uij + u[idx - 1]) * invdx2 +
        (u[idx + nx] - 2.0f * uij + u[idx - nx]) * invdy2;

    // Accumulate source contributions
    float source_acc = 0.0f;
    float constant = -1.0f;

    const float x = i * dx;
    const float y = j * dy;

    for (std::size_t k = 0; k < num_sources; ++k)
    {
        const auto& s = sources[k];
        const float val = device_source_value_at(s, t_mid, x, y, dt, dx, dy);

        if (s.temporal_kind == Source::TemporalKind::Rate)
        {
            source_acc += val;
        }
        else if (s.temporal_kind == Source::TemporalKind::Constant)
        {
            constant = fmaxf(constant, val);
        }
    }

    const float aij = alpha[idx];

    if (constant > 0.0f)
    {
        un[idx] = constant;
    }
    else
    {
        un[idx] = uij + aij * dt * (lap + source_acc);
    }
}

// -------------------- Private helpers --------------------

void CudaEngine::_check_data_dims(const Grid& model_grid) const
{
    const auto n {nx * ny};

    if (model_grid.nx != nx || model_grid.ny != ny)
    {
        throw std::runtime_error("DeviceGrid: Grid dims mismatch");
    }

    if (model_grid.u.size() != n ||
        model_grid.un.size() != n ||
        model_grid.alpha.size() != n)
    {
        throw std::runtime_error("DeviceGrid: Grid vector sizes mismatch");
    }
}

void CudaEngine::_check_sources_dims(const std::vector<Source>& sources) const
{
    if (sources.size() != num_sources)
    {
        throw std::runtime_error("DeviceGrid: Sources size mismatch");
    }
}

// -------------------- Lifecycle --------------------

void CudaEngine::_allocate()
{
    // ---- scalar/device attributes ----
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nx), sizeof(std::size_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ny), sizeof(std::size_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_num_sources), sizeof(std::size_t)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dx), sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dy), sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_length_x), sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_length_y), sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_invdx2), sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_invdy2), sizeof(float)));

    // ---- arrays ----
    const std::size_t bytes = nx * ny * sizeof(float);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_u), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_un), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_alpha), bytes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sources),
                          num_sources * sizeof(Source)));
}

void CudaEngine::_cleanup()
{
    // arrays
    if (d_u)       { cudaFree(d_u); d_u = nullptr; }
    if (d_un)      { cudaFree(d_un); d_un = nullptr; }
    if (d_alpha)   { cudaFree(d_alpha); d_alpha = nullptr; }
    if (d_sources) { cudaFree(d_sources); d_sources = nullptr; }

    // scalars
    if (d_dx)        { cudaFree(d_dx); d_dx = nullptr; }
    if (d_dy)        { cudaFree(d_dy); d_dy = nullptr; }
    if (d_length_x)  { cudaFree(d_length_x); d_length_x = nullptr; }
    if (d_length_y)  { cudaFree(d_length_y); d_length_y = nullptr; }
    if (d_invdx2)    { cudaFree(d_invdx2); d_invdx2 = nullptr; }
    if (d_invdy2)    { cudaFree(d_invdy2); d_invdy2 = nullptr; }

    if (d_nx)          { cudaFree(d_nx); d_nx = nullptr; }
    if (d_ny)          { cudaFree(d_ny); d_ny = nullptr; }
    if (d_num_sources) { cudaFree(d_num_sources); d_num_sources = nullptr; }
}

// Copy model grid from host -> device
void CudaEngine::upload_grid(const Grid& model_grid,
                             const std::vector<Source>& sources)
{
    // Validate sizes
    _check_data_dims(model_grid);
    _check_sources_dims(sources);

    const auto n {model_grid.nx * model_grid.ny};
    const auto bytes {n * sizeof(float)};
    const auto num_sources_host {sources.size()};

    // Copy scalar attributes
    CUDA_CHECK(cudaMemcpy(d_nx, &model_grid.nx, sizeof(std::size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ny, &model_grid.ny, sizeof(std::size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dx, &model_grid.dx, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dy, &model_grid.dy, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_length_x, &model_grid.length_x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_length_y, &model_grid.length_y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_invdx2, &model_grid.invdx2, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_invdy2, &model_grid.invdy2, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_sources, &num_sources_host, sizeof(std::size_t), cudaMemcpyHostToDevice));

    // Copy arrays
    CUDA_CHECK(cudaMemcpy(d_u,     model_grid.u.data(),     bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_un,    model_grid.un.data(),    bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alpha, model_grid.alpha.data(), bytes, cudaMemcpyHostToDevice));

    // Copy sources (ensure Source is trivially copyable / POD)
    if (num_sources_host)
    {
        CUDA_CHECK(cudaMemcpy(d_sources, sources.data(),
                              num_sources_host * sizeof(Source),
                              cudaMemcpyHostToDevice));
    }
}

// Copy important data from device -> host
void CudaEngine::download_grid(Grid& model_grid)
{
    _check_data_dims(model_grid);

    const auto n {model_grid.nx * model_grid.ny};
    const auto bytes {n * sizeof(float)};

    CUDA_CHECK(cudaMemcpy(model_grid.u.data(),  d_u,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(model_grid.un.data(), d_un, bytes, cudaMemcpyDeviceToHost));
}

// -------------------- Ops --------------------

void CudaEngine::dirichlet_boundaries(cudaStream_t stream)
{
    const auto NX {nx};
    const auto NY {ny};

    if (NX == 0 || NY == 0) return;

    const int threads = 256;
    const int blocks_x = static_cast<int>((NX + threads - 1) / threads);
    const int blocks_y = static_cast<int>((NY + threads - 1) / threads);

    set_top_bottom<<<blocks_x, threads, 0, stream>>>(d_un, NX, NY);
    set_left_right<<<blocks_y, threads, 0, stream>>>(d_un, NX, NY);

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CudaEngine::swap_buffers() noexcept
{
    if (d_u == nullptr || d_un == nullptr)
    {
        return;
    }

    std::swap(d_u, d_un);
}

void CudaEngine::heat_step(float dt, float t_mid)
{
    if (nx < 2 || ny < 2) return;

    // pull scalar grid params from device (or cache these on host later)
    float h_invdx2{}, h_invdy2{}, h_dx{}, h_dy{};
    CUDA_CHECK(cudaMemcpy(&h_invdx2, d_invdx2, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_invdy2, d_invdy2, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_dx,     d_dx,     sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_dy,     d_dy,     sizeof(float), cudaMemcpyDeviceToHost));

    dim3 block(16, 16);
    dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y);

    heat_step_kernel<<<grid, block>>>(
        d_u, d_un, d_alpha,
        nx, ny,
        h_invdx2, h_invdy2,
        h_dx, h_dy, dt,
        d_sources, num_sources,
        t_mid
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // keep for bring-up; remove later if you pipeline
}
