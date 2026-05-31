#ifndef OPENCLENGINE_HPP
#define OPENCLENGINE_HPP

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "heat/Grid.hpp"
#include "heat/Source.hpp"

struct OpenCLEngine {
    cl_platform_id platform {};
    cl_device_id device {};
    cl_context context {};
    cl_command_queue queue {};

    cl_program program {};
    cl_kernel heat_kernel {};
    cl_kernel boundary_tb_kernel {};
    cl_kernel boundary_lr_kernel {};

    std::size_t nx {};
    std::size_t ny {};
    std::size_t num_sources {};

    cl_mem d_u {};
    cl_mem d_un {};
    cl_mem d_alpha {};
    cl_mem d_sources {};
    cl_mem d_num_sources {};

    OpenCLEngine(const std::string& device_name,
                 std::size_t nx_in,
                 std::size_t ny_in,
                 std::size_t num_sources_in)
        : nx(nx_in),
          ny(ny_in),
          num_sources(num_sources_in)
    {
        _init_opencl(device_name);
        _build_program();
        _allocate();
    }

    ~OpenCLEngine()
    {
        _cleanup();
    }

    void upload_grid(const Grid& model_grid, const std::vector<Source>& sources)
    {
        _check_data_dims(model_grid);
        _check_sources_dims(sources);

        const std::size_t n = nx * ny;

        clEnqueueWriteBuffer(queue,
                             d_u,
                             CL_TRUE,
                             0,
                             sizeof(double) * n,
                             model_grid.u.data(),
                             0,
                             nullptr,
                             nullptr);

        clEnqueueWriteBuffer(queue,
                             d_un,
                             CL_TRUE,
                             0,
                             sizeof(double) * n,
                             model_grid.un.data(),
                             0,
                             nullptr,
                             nullptr);

        clEnqueueWriteBuffer(queue,
                             d_alpha,
                             CL_TRUE,
                             0,
                             sizeof(double) * n,
                             model_grid.alpha.data(),
                             0,
                             nullptr,
                             nullptr);

                             num_sources = sources.size();

        clEnqueueWriteBuffer(queue,
                            d_num_sources,
                            CL_TRUE,
                            0,
                            sizeof(std::size_t),
                            &num_sources,
                            0,
                            nullptr,
                            nullptr);

        if (num_sources > 0) {
            clEnqueueWriteBuffer(queue,
                                d_sources,
                                CL_TRUE,
                                0,
                                sizeof(Source) * num_sources,
                                sources.data(),
                                0,
                                nullptr,
                                nullptr);
        }
    }

    void download_grid(Grid& model_grid)
    {
        _check_data_dims(model_grid);

        const std::size_t n = nx * ny;

        clEnqueueReadBuffer(queue,
                            d_u,
                            CL_TRUE,
                            0,
                            sizeof(double) * n,
                            model_grid.u.data(),
                            0,
                            nullptr,
                            nullptr);

        clEnqueueReadBuffer(queue,
                            d_un,
                            CL_TRUE,
                            0,
                            sizeof(double) * n,
                            model_grid.un.data(),
                            0,
                            nullptr,
                            nullptr);
    }

    void dirichlet_boundaries()
    {
        const cl_ulong NX = nx;
        const cl_ulong NY = ny;

        clSetKernelArg(boundary_tb_kernel, 0, sizeof(cl_mem), &d_un);
        clSetKernelArg(boundary_tb_kernel, 1, sizeof(cl_ulong), &NX);
        clSetKernelArg(boundary_tb_kernel, 2, sizeof(cl_ulong), &NY);

        size_t global_tb[1] = {NX};

        clEnqueueNDRangeKernel(queue,
                               boundary_tb_kernel,
                               1,
                               nullptr,
                               global_tb,
                               nullptr,
                               0,
                               nullptr,
                               nullptr);

        clSetKernelArg(boundary_lr_kernel, 0, sizeof(cl_mem), &d_un);
        clSetKernelArg(boundary_lr_kernel, 1, sizeof(cl_ulong), &NX);
        clSetKernelArg(boundary_lr_kernel, 2, sizeof(cl_ulong), &NY);

        size_t global_lr[1] = {NY};

        clEnqueueNDRangeKernel(queue,
                               boundary_lr_kernel,
                               1,
                               nullptr,
                               global_lr,
                               nullptr,
                               0,
                               nullptr,
                               nullptr);

        clFinish(queue);
    }

    void heat_step(const Grid& grid, double dt, double t_sample)
    {
        const cl_ulong NX = nx;
        const cl_ulong NY = ny;

        const double invdx2 = grid.invdx2;
        const double invdy2 = grid.invdy2;

        cl_ulong source_count = num_sources;

        clSetKernelArg(heat_kernel, 0, sizeof(cl_ulong), &NX);
        clSetKernelArg(heat_kernel, 1, sizeof(cl_ulong), &NY);
        clSetKernelArg(heat_kernel, 2, sizeof(double), &dt);
        clSetKernelArg(heat_kernel, 3, sizeof(double), &t_sample);
        clSetKernelArg(heat_kernel, 4, sizeof(double), &grid.invdx2);
        clSetKernelArg(heat_kernel, 5, sizeof(double), &grid.invdy2);
        clSetKernelArg(heat_kernel, 6, sizeof(double), &grid.dx);
        clSetKernelArg(heat_kernel, 7, sizeof(double), &grid.dy);
        clSetKernelArg(heat_kernel, 8, sizeof(cl_mem), &d_u);
        clSetKernelArg(heat_kernel, 9, sizeof(cl_mem), &d_un);
        clSetKernelArg(heat_kernel,10, sizeof(cl_mem), &d_alpha);
        clSetKernelArg(heat_kernel,11, sizeof(cl_mem), &d_sources);
        clSetKernelArg(heat_kernel,12, sizeof(cl_ulong), &source_count);

        size_t global[2] = {NY - 2, NX - 2};

        clEnqueueNDRangeKernel(queue,
                               heat_kernel,
                               2,
                               nullptr,
                               global,
                               nullptr,
                               0,
                               nullptr,
                               nullptr);

        clFinish(queue);
    }

    void swap_buffers()
    {
        std::swap(d_u, d_un);
    }

private:
    void _init_opencl(const std::string& device_preference)
    {
        cl_int err {};

        cl_uint num_platforms {};
        clGetPlatformIDs(0, nullptr, &num_platforms);

        if (num_platforms == 0) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        platform = platforms[0];

        cl_device_type dtype = CL_DEVICE_TYPE_DEFAULT;

        if (device_preference == "cpu") {
            dtype = CL_DEVICE_TYPE_CPU;
        } else if (device_preference == "gpu") {
            dtype = CL_DEVICE_TYPE_GPU;
        }

        cl_uint num_devices {};
        clGetDeviceIDs(platform, dtype, 0, nullptr, &num_devices);

        if (num_devices == 0) {
            throw std::runtime_error("No matching OpenCL device");
        }

        std::vector<cl_device_id> devices(num_devices);

        clGetDeviceIDs(platform,
                       dtype,
                       num_devices,
                       devices.data(),
                       nullptr);

        device = devices[0];

        context = clCreateContext(nullptr,
                                  1,
                                  &device,
                                  nullptr,
                                  nullptr,
                                  &err);

        queue = clCreateCommandQueue(context,
                                     device,
                                     0,
                                     &err);

        char name[256];

        clGetDeviceInfo(device,
                        CL_DEVICE_NAME,
                        sizeof(name),
                        name,
                        nullptr);

        std::cerr << "Using OpenCL device: " << name << "\n";
    }

    void _build_program()
    {
        const char* source = R"CLC(

        #pragma OPENCL EXTENSION cl_khr_fp64 : enable

        typedef struct
        {
            int spatial_kind;
            int temporal_kind;

            double t0;
            double duration;
            double amplitude;

            double x0;
            double y0;
            double sigma;

            double x_min;
            double x_max;
            double y_min;
            double y_max;

        } Source;

        inline double source_value_at_device(
            Source s,
            double t,
            double x,
            double y,
            double dt,
            double dx,
            double dy)
        {
            int active = 0;

            /* Constant = 1, Rate = 2, Impulse = 3 */

            if (s.temporal_kind == 1 ||
                s.temporal_kind == 2)
            {
                active =
                    (t >= s.t0) &&
                    (t < (s.t0 + s.duration));
            }
            else if (s.temporal_kind == 3)
            {
                active =
                    (t >= s.t0) &&
                    (t < (s.t0 + dt));
            }

            if (!active)
                return 0.0;

            double spatial = 0.0;

            /* Gaussian = 1 */
            if (s.spatial_kind == 1)
            {
                const double dx0 = x - s.x0;
                const double dy0 = y - s.y0;

                const double two_sigma2 =
                    2.0 * s.sigma * s.sigma + 1e-300;

                spatial =
                    exp(-(dx0 * dx0 + dy0 * dy0) /
                        two_sigma2);
            }

            /* Point = 2 */
            else if (s.spatial_kind == 2)
            {
                const double hx = 0.5 * dx;
                const double hy = 0.5 * dy;

                spatial =
                    (fabs(x - s.x0) <= hx &&
                    fabs(y - s.y0) <= hy)
                        ? 1.0
                        : 0.0;
            }

            /* Block = 3 */
            else if (s.spatial_kind == 3)
            {
                spatial =
                    (x >= s.x_min &&
                    x <= s.x_max &&
                    y >= s.y_min &&
                    y <= s.y_max)
                        ? 1.0
                        : 0.0;
            }

            return s.amplitude * spatial;
        }

        __kernel void heat_step(
            const ulong NX,
            const ulong NY,
            const double dt,
            const double t_sample,
            const double invdx2,
            const double invdy2,
            const double dx,
            const double dy,
            __global const double* u,
            __global double* un,
            __global const double* alpha,
            __global const Source* sources,
            const ulong source_count)
        {
            const size_t j = get_global_id(0) + 1;
            const size_t i = get_global_id(1) + 1;

            if (i >= NX - 1 ||
                j >= NY - 1)
            {
                return;
            }

            const size_t idx = j * NX + i;

            const double x = i * dx;
            const double y = j * dy;

            const double uij = u[idx];

            const double lap =
                (u[idx + 1] - 2.0 * uij + u[idx - 1]) * invdx2 +
                (u[idx + NX] - 2.0 * uij + u[idx - NX]) * invdy2;

            double source_acc = 0.0;
            double constant_source = -1.0;

            for (ulong k = 0; k < source_count; ++k)
            {
                const Source s = sources[k];

                const double val =
                    source_value_at_device(
                        s,
                        t_sample,
                        x,
                        y,
                        dt,
                        dx,
                        dy);

                /* constant_source */
                if (s.temporal_kind == 1)
                {
                    constant_source = fmax(constant_source, val);
                }

                /* Rate */
                else if (s.temporal_kind == 2)
                {
                    source_acc += val;
                }

                /* Impulse */
                else if (s.temporal_kind == 3)
                {
                    source_acc += val;
                }
            }

            const double aij = alpha[idx];

            if (constant_source > 0.0)
            {
                un[idx] = constant_source;
            }
            else
            {
                un[idx] =
                    uij +
                    aij * dt *
                    (lap + source_acc);
            }
        }

        __kernel void boundary_tb(
            __global double* un,
            const ulong NX,
            const ulong NY)
        {
            const size_t i = get_global_id(0);

            un[i] = 0.0;
            un[(NY - 1) * NX + i] = 0.0;
        }

        __kernel void boundary_lr(
            __global double* un,
            const ulong NX,
            const ulong NY)
        {
            const size_t j = get_global_id(0);

            un[j * NX] = 0.0;
            un[j * NX + (NX - 1)] = 0.0;
        }

        )CLC";

        cl_int err {};

        program = clCreateProgramWithSource(context,
                                            1,
                                            &source,
                                            nullptr,
                                            &err);

        err = clBuildProgram(program,
                             1,
                             &device,
                             "",
                             nullptr,
                             nullptr);

        if (err != CL_SUCCESS) {

            size_t log_size {};
            clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_LOG,
                                  0,
                                  nullptr,
                                  &log_size);

            std::vector<char> log(log_size);

            clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_LOG,
                                  log_size,
                                  log.data(),
                                  nullptr);

            std::cerr << log.data() << "\n";

            throw std::runtime_error("OpenCL build failed");
        }

        heat_kernel =
            clCreateKernel(program, "heat_step", &err);

        boundary_tb_kernel =
            clCreateKernel(program, "boundary_tb", &err);

        boundary_lr_kernel =
            clCreateKernel(program, "boundary_lr", &err);
    }

    void _allocate()
    {
        cl_int err {};

        const std::size_t n = nx * ny;

        d_u = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             sizeof(double) * n,
                             nullptr,
                             &err);

        d_un = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              sizeof(double) * n,
                              nullptr,
                              &err);

        d_alpha = clCreateBuffer(context,
                                 CL_MEM_READ_WRITE,
                                 sizeof(double) * n,
                                 nullptr,
                                 &err);

        d_num_sources = clCreateBuffer(context,
                        CL_MEM_READ_ONLY,
                        sizeof(std::size_t),
                        nullptr,
                        &err);

        const std::size_t nsrc_alloc =
            std::max<std::size_t>(1, num_sources);

        d_sources =
            clCreateBuffer(context,
                        CL_MEM_READ_ONLY,
                        sizeof(Source) * nsrc_alloc,
                        nullptr,
                        &err);

        if (!d_u || !d_un || !d_alpha) {
            throw std::runtime_error("OpenCL allocation failed");
        }
    }

    void _cleanup()
    {
        if (d_u)
            clReleaseMemObject(d_u);

        if (d_un)
            clReleaseMemObject(d_un);

        if (d_alpha)
            clReleaseMemObject(d_alpha);

        if (d_sources)
            clReleaseMemObject(d_sources);

        if (d_num_sources)
            clReleaseMemObject(d_num_sources);

        if (heat_kernel)
            clReleaseKernel(heat_kernel);

        if (boundary_tb_kernel)
            clReleaseKernel(boundary_tb_kernel);

        if (boundary_lr_kernel)
            clReleaseKernel(boundary_lr_kernel);

        if (program)
            clReleaseProgram(program);

        if (queue)
            clReleaseCommandQueue(queue);

        if (context)
            clReleaseContext(context);
    }

    void _check_data_dims(const Grid& model_grid) const
    {
        const auto n = nx * ny;

        if (model_grid.nx != nx ||
            model_grid.ny != ny) {
            throw std::runtime_error("Grid dims mismatch");
        }

        if (model_grid.u.size() != n ||
            model_grid.un.size() != n ||
            model_grid.alpha.size() != n) {
            throw std::runtime_error("Grid vector size mismatch");
        }
    }

    void _check_sources_dims(const std::vector<Source>& sources) const
    {
        if (sources.size() != num_sources) {
            throw std::runtime_error("Sources size mismatch");
        }
    }
};

#endif