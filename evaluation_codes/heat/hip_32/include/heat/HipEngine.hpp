#ifndef HIPENGINE_HPP
#define HIPENGINE_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include "heat/Grid.hpp"
#include "heat/HipFunctions.hpp"
#include "heat/Source.hpp"

inline void hip_check_impl(hipError_t err, const char* call, const char* file, int line) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error: ") + call + " failed at " + file + ":" +
                                 std::to_string(line) + " with " + hipGetErrorString(err));
    }
}

#define HIP_CHECK(call) hip_check_impl((call), #call, __FILE__, __LINE__)

[[nodiscard]] __host__ __device__ inline float source_value_at_device(
    const Source& s, float t, float x, float y, float dt, float dx, float dy) {
    bool active = false;
    switch (s.temporal_kind) {
        case Source::TemporalKind::Constant:
        case Source::TemporalKind::Rate:
            active = (t >= s.t0) && (t < s.t0 + s.duration);
            break;
        case Source::TemporalKind::Impulse:
            active = (t >= s.t0) && (t < s.t0 + dt);
            break;
    }

    if (!active) {
        return 0.0f;
    }

    float spatial = 0.0f;
    switch (s.spatial_kind) {
        case Source::SpatialKind::Gaussian: {
            const float dx0 = x - s.x0;
            const float dy0 = y - s.y0;
            const float two_sigma2 = 2.0f * s.sigma * s.sigma + 1e-38f;
            spatial = Maths::exp(-(dx0 * dx0 + dy0 * dy0) / two_sigma2);
            break;
        }
        case Source::SpatialKind::Block:
            spatial = (x >= s.x_min && x <= s.x_max && y >= s.y_min && y <= s.y_max) ? 1.0f : 0.0f;
            break;
        case Source::SpatialKind::Point: {
            const float hx = 0.5f * dx;
            const float hy = 0.5f * dy;
            spatial = (Maths::fabs(x - s.x0) <= hx && Maths::fabs(y - s.y0) <= hy) ? 1.0f : 0.0f;
            break;
        }
    }

    return s.amplitude * spatial;
}

__global__ void dirichlet_rows_kernel(float* un, std::size_t nx, std::size_t ny) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) {
        return;
    }

    un[i] = 0.0f;
    un[(ny - 1) * nx + i] = 0.0f;
}

__global__ void dirichlet_cols_kernel(float* un, std::size_t nx, std::size_t ny) {
    const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) {
        return;
    }

    un[j * nx] = 0.0f;
    un[j * nx + (nx - 1)] = 0.0f;
}

__global__ void heat_step_kernel(std::size_t nx, std::size_t ny, float dx, float dy,
                                 float invdx2, float invdy2, const float* u, float* un,
                                 const float* alpha, const Source* sources,
                                 std::size_t num_sources, float dt, float t_sample) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const std::size_t j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= nx - 1 || j >= ny - 1) {
        return;
    }

    const std::size_t idx = j * nx + i;

    const float x = static_cast<float>(i) * dx;
    const float y = static_cast<float>(j) * dy;

    const float uij = u[idx];
    const float lap = (u[idx + 1] - 2.0f * uij + u[idx - 1]) * invdx2 +
                       (u[idx + nx] - 2.0f * uij + u[idx - nx]) * invdy2;

    float source_accumulator = 0.0f;
    float constant = -1.0f;

    for (std::size_t k = 0; k < num_sources; ++k) {
        const Source s = sources[k];
        const float val = source_value_at_device(s, t_sample, x, y, dt, dx, dy);

        if (s.temporal_kind == Source::TemporalKind::Rate) {
            source_accumulator += val;
        } else if (s.temporal_kind == Source::TemporalKind::Constant) {
            constant = Maths::fmax(constant, val);
        } else {
            source_accumulator += val;
        }
    }

    const float aij = alpha ? alpha[idx] : 1.0f;
    un[idx] = (constant > 0.0f) ? constant : (uij + aij * dt * (lap + source_accumulator));
}

struct HipEngine {
    std::size_t nx{};
    std::size_t ny{};
    std::size_t num_sources{};
    std::size_t source_capacity{};

    float dx{};
    float dy{};
    float length_x{};
    float length_y{};
    float invdx2{};
    float invdy2{};

    float* d_u{nullptr};
    float* d_un{nullptr};
    float* d_alpha{nullptr};
    Source* d_sources{nullptr};

    HipEngine(const std::string& device, std::size_t nx_in, std::size_t ny_in,
              std::size_t num_sources_in)
        : nx(nx_in), ny(ny_in), num_sources(num_sources_in),
          source_capacity(std::max<std::size_t>(1, num_sources_in)) {
        _make_device(device);
        _allocate();
    }

    ~HipEngine() {
        _cleanup();
    }

    HipEngine(const HipEngine&) = delete;
    HipEngine& operator=(const HipEngine&) = delete;

    void upload_grid(const Grid& model_grid, const std::vector<Source>& sources) {
        _check_data_dims(model_grid);
        _check_sources_dims(sources);

        const auto n = model_grid.nx * model_grid.ny;

        nx = model_grid.nx;
        ny = model_grid.ny;
        dx = model_grid.dx;
        dy = model_grid.dy;
        length_x = model_grid.length_x;
        length_y = model_grid.length_y;
        invdx2 = model_grid.invdx2;
        invdy2 = model_grid.invdy2;

        HIP_CHECK(hipMemcpy(d_u, model_grid.u.data(), sizeof(float) * n, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_un, model_grid.un.data(), sizeof(float) * n, hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(d_alpha, model_grid.alpha.data(), sizeof(float) * n, hipMemcpyHostToDevice));

        num_sources = sources.size();
        if (num_sources > 0) {
            HIP_CHECK(hipMemcpy(d_sources, sources.data(), sizeof(Source) * num_sources,
                                hipMemcpyHostToDevice));
        }
    }

    void download_grid(Grid& model_grid) {
        _check_data_dims(model_grid);

        const auto n = model_grid.nx * model_grid.ny;

        HIP_CHECK(hipMemcpy(model_grid.u.data(), d_u, sizeof(float) * n, hipMemcpyDeviceToHost));
        HIP_CHECK(
            hipMemcpy(model_grid.un.data(), d_un, sizeof(float) * n, hipMemcpyDeviceToHost));
    }

    void dirichlet_boundaries() {
        constexpr std::size_t block_size = 256;

        const auto row_blocks = static_cast<unsigned int>((nx + block_size - 1) / block_size);
        hipLaunchKernelGGL(dirichlet_rows_kernel, dim3(row_blocks), dim3(block_size), 0, 0, d_un, nx,
                           ny);
        HIP_CHECK(hipGetLastError());

        const auto col_blocks = static_cast<unsigned int>((ny + block_size - 1) / block_size);
        hipLaunchKernelGGL(dirichlet_cols_kernel, dim3(col_blocks), dim3(block_size), 0, 0, d_un, nx,
                           ny);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }

    void heat_step(float dt, float t_sample) {
        if (nx < 3 || ny < 3) {
            return;
        }

        constexpr unsigned int block_x = 16;
        constexpr unsigned int block_y = 16;
        const dim3 block(block_x, block_y);
        const dim3 grid(static_cast<unsigned int>((nx - 2 + block_x - 1) / block_x),
                        static_cast<unsigned int>((ny - 2 + block_y - 1) / block_y));

        hipLaunchKernelGGL(heat_step_kernel, grid, block, 0, 0, nx, ny, dx, dy, invdx2, invdy2, d_u,
                           d_un, d_alpha, d_sources, num_sources, dt, t_sample);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }

    void swap_buffers() {
        std::swap(d_u, d_un);
    }

    [[nodiscard]] std::size_t size() const {
        return nx * ny;
    }

private:
    static bool _is_unsigned_integer(const std::string& text) {
        return !text.empty() &&
               std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c); });
    }

    void _make_device(const std::string& device) {
        int device_count = 0;
        HIP_CHECK(hipGetDeviceCount(&device_count));
        if (device_count < 1) {
            throw std::runtime_error("HIP backend did not find a GPU device");
        }

        auto device_id = 0;
        if (_is_unsigned_integer(device)) {
            device_id = std::stoi(device);
        } else if (device == "cpu") {
            std::cerr << "Warning: HIP backend requires a GPU; using HIP device 0 instead of CPU.\n";
        } else if (device != "gpu" && device != "default" && !device.empty()) {
            std::cerr << "Warning: unknown HIP device selector \"" << device
                      << "\"; using HIP device 0.\n";
        }

        if (device_id < 0 || device_id >= device_count) {
            throw std::runtime_error("Requested HIP device id is out of range");
        }

        HIP_CHECK(hipSetDevice(device_id));

        hipDeviceProp_t props{};
        HIP_CHECK(hipGetDeviceProperties(&props, device_id));
        std::cerr << "Using HIP device " << device_id << ": " << props.name << "\n";
    }

    void _allocate() {
        if (nx == 0 || ny == 0) {
            throw std::runtime_error("HipEngine: zero dimensions");
        }

        const auto n = nx * ny;

        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_u), sizeof(float) * n));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_un), sizeof(float) * n));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_alpha), sizeof(float) * n));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_sources), sizeof(Source) * source_capacity));
    }

    void _cleanup() noexcept {
        if (d_u) {
            hipFree(d_u);
            d_u = nullptr;
        }
        if (d_un) {
            hipFree(d_un);
            d_un = nullptr;
        }
        if (d_alpha) {
            hipFree(d_alpha);
            d_alpha = nullptr;
        }
        if (d_sources) {
            hipFree(d_sources);
            d_sources = nullptr;
        }
    }

    void _check_data_dims(const Grid& model_grid) const {
        const auto n = nx * ny;
        if (model_grid.nx != nx || model_grid.ny != ny) {
            throw std::runtime_error("HipEngine: Grid dims mismatch");
        }
        if (model_grid.u.size() != n || model_grid.un.size() != n || model_grid.alpha.size() != n) {
            throw std::runtime_error("HipEngine: Grid vector sizes mismatch");
        }
    }

    void _check_sources_dims(const std::vector<Source>& sources) const {
        if (sources.size() > source_capacity) {
            throw std::runtime_error("HipEngine: Sources size exceeds allocated capacity");
        }
    }
};

#endif  // HIPENGINE_HPP
