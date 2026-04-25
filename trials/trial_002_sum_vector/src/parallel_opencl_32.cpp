/**
 * @file serial.cpp
 * @brief
 *
 * @author Ben Palmer
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026 Ben Palmer
 * SPDX-License-Identifier: MIT
 */

#define CL_TARGET_OPENCL_VERSION 120
#ifndef CL_PLATFORM_NOT_FOUND_KHR
#define CL_PLATFORM_NOT_FOUND_KHR -1001
#endif

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <system_error>
#include <vector>
#include <utility>

#include "helper/Error.hpp"
#include "helper/helper.hpp"

#include <nlohmann/json.hpp>

// ======================================================
// Serial baseline (unchanged)
// ======================================================

float serial_naive_task(const std::vector<float>& numbers) {
    float sum{0.0f};
    for (const auto val : numbers)
        sum += val;
    return sum;
}

void opencl_check(cl_int status, const char* message) {
    if (status != CL_SUCCESS) {
        std::ostringstream oss;
        oss << message << " (OpenCL error " << status << ")";
        THROW_RUNTIME_ERROR(oss.str());
    }
}

// ======================================================
// OpenCL kernel
// ======================================================

const char* opencl_kernel_code = R"CLC(
__kernel void reduce_sum(__global const float* input,
                         __global float* output,
                         __local float* local_mem,
                         const int N)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int gsize = get_global_size(0);
    int group_size = get_local_size(0);

    float sum = 0.0f;

    int N4 = N / 4;
    __global const float4* input4 = (__global const float4*)input;

    for (int i = gid; i < N4; i += gsize) {
        float4 v = input4[i];
        sum += v.x + v.y + v.z + v.w;
    }

    for (int i = N4 * 4 + gid; i < N; i += gsize) {
        sum += input[i];
    }

    local_mem[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (group_size >= 256) { if (lid < 128) local_mem[lid] += local_mem[lid + 128]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (group_size >= 128) { if (lid < 64)  local_mem[lid] += local_mem[lid + 64];  barrier(CLK_LOCAL_MEM_FENCE); }
    if (group_size >= 64)  { if (lid < 32)  local_mem[lid] += local_mem[lid + 32];  barrier(CLK_LOCAL_MEM_FENCE); }

    if (lid < 32) {
        local_mem[lid] += local_mem[lid + 32];
        local_mem[lid] += local_mem[lid + 16];
        local_mem[lid] += local_mem[lid + 8];
        local_mem[lid] += local_mem[lid + 4];
        local_mem[lid] += local_mem[lid + 2];
        local_mem[lid] += local_mem[lid + 1];
    }

    if (lid == 0)
        output[get_group_id(0)] = local_mem[0];
}
)CLC";

// OpenCL context
//======================================================

struct OpenCLContext {
    cl_platform_id platform{};
    cl_device_id device{};
    cl_context context{};
    cl_command_queue queue{};
    cl_program program{};
    cl_kernel kernel{};

    cl_mem input_buf{};
    cl_mem buffer_a{};
    cl_mem buffer_b{};

    size_t local_size{256};
};

// Setup
//======================================================

void opencl_setup(OpenCLContext& ctx, const std::vector<float>& numbers,
                  std::string_view device_string) {
    cl_int err;

    // ---------------------------
    // 1. Get platforms
    // ---------------------------
    cl_uint platform_count{0};
    cl_int status = clGetPlatformIDs(0, nullptr, &platform_count);

    if (status == CL_PLATFORM_NOT_FOUND_KHR || platform_count == 0) {
        THROW_RUNTIME_ERROR("No OpenCL platforms found.");
    }

    std::vector<cl_platform_id> platforms(platform_count);
    opencl_check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
                 "clGetPlatformIDs failed");

    // ---------------------------
    // 2. Select device type
    // ---------------------------
    const cl_device_type requested_type =
        (device_string == "CPU") ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;

    // ---------------------------
    // 3. Find device
    // ---------------------------
    ctx.device = nullptr;
    ctx.platform = nullptr;

    for (const auto& platform : platforms) {
        cl_uint device_count{0};
        cl_int device_status = clGetDeviceIDs(platform, requested_type, 0, nullptr, &device_count);

        if (device_status == CL_SUCCESS && device_count > 0) {
            std::vector<cl_device_id> devices(device_count);
            opencl_check(
                clGetDeviceIDs(platform, requested_type, device_count, devices.data(), nullptr),
                "clGetDeviceIDs failed");

            ctx.device = devices.front();
            ctx.platform = platform;
            break;
        }
    }

    if (!ctx.device) {
        std::ostringstream oss;
        oss << "No suitable OpenCL " << device_string << " device found.";
        THROW_RUNTIME_ERROR(oss.str());
    }

    // ---------------------------
    // 4. Create context + queue
    // ---------------------------
    ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &err);
    opencl_check(err, "clCreateContext failed");

#if CL_TARGET_OPENCL_VERSION >= 200
    ctx.queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, 0, &err);
#else
    ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, &err);
#endif
    opencl_check(err, "clCreateCommandQueueWithProperties failed");

    // ---------------------------
    // 5. Build program
    // ---------------------------
    const char* src = opencl_kernel_code;

    ctx.program = clCreateProgramWithSource(ctx.context, 1, &src, nullptr, &err);
    opencl_check(err, "clCreateProgramWithSource failed");

    err = clBuildProgram(ctx.program, 1, &ctx.device, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        // Optional: print build log (VERY useful)
        size_t log_size = 0;
        clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        std::vector<char> log(log_size);
        clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                              nullptr);

        std::cerr << "Build log:\n" << log.data() << "\n";
        opencl_check(err, "clBuildProgram failed");
    }

    ctx.kernel = clCreateKernel(ctx.program, "reduce_sum", &err);
    opencl_check(err, "clCreateKernel failed");

    // ---------------------------
    // 6. Buffers (OPTIMISED)
    // ---------------------------
    const int N = numbers.size();

    // tune local size
    size_t max_workgroup = 0;
    clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup,
                    nullptr);

    ctx.local_size = std::min<size_t>(256, max_workgroup);

    // compute max number of groups
    size_t max_groups = (N + ctx.local_size - 1) / ctx.local_size;

    ctx.input_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * N, const_cast<float*>(numbers.data()), &err);
    opencl_check(err, "clCreateBuffer input_buf failed");

    ctx.buffer_a =
        clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * max_groups, nullptr, &err);
    opencl_check(err, "clCreateBuffer buffer_a failed");

    ctx.buffer_b =
        clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * max_groups, nullptr, &err);
    opencl_check(err, "clCreateBuffer buffer_b failed");
}

// Device reduction
//======================================================

float parallel_task(OpenCLContext& ctx, int N) {
    cl_mem in = ctx.input_buf;
    cl_mem out = ctx.buffer_a;

    int current_N = N;
    size_t local_size = ctx.local_size;

    while (current_N > 1) {
        size_t global_size = ((current_N + local_size - 1) / local_size) * local_size;

        size_t num_groups = global_size / local_size;

        clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &in);
        clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &out);
        clSetKernelArg(ctx.kernel, 2, sizeof(float) * local_size, nullptr);
        clSetKernelArg(ctx.kernel, 3, sizeof(int), &current_N);

        clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr, &global_size, &local_size, 0,
                               nullptr, nullptr);

        current_N = num_groups;

        std::swap(in, out);
        out = (out == ctx.buffer_a) ? ctx.buffer_b : ctx.buffer_a;
    }

    float result{};
    clEnqueueReadBuffer(ctx.queue, in, CL_TRUE, 0, sizeof(float), &result, 0, nullptr, nullptr);

    return result;
}

// Cleanup (t2 to t3)
//======================================================

void opencl_cleanup(OpenCLContext& ctx) {
    clReleaseMemObject(ctx.input_buf);
    clReleaseMemObject(ctx.buffer_a);
    clReleaseMemObject(ctx.buffer_b);

    clReleaseKernel(ctx.kernel);
    clReleaseProgram(ctx.program);
    clReleaseCommandQueue(ctx.queue);
    clReleaseContext(ctx.context);
}

// MAIN
//======================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit vec_size\n";
        return 1;
    }

    const double test_time_seconds = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);

    std::string_view device_string = "GPU";
    if (argc >= 4) {
        device_string = argv[3];

        if (device_string != "GPU" && device_string != "CPU") {
            THROW_INVALID_ARGUMENT("device must be GPU or CPU");
        }
    }

    // Random data
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<float> numbers;
    numbers.reserve(N);

    for (int i = 0; i < N; ++i) {
        numbers.emplace_back(static_cast<float>(dist(rng)));
    }

    // Timing
    //==================================================

    auto t0 = std::chrono::steady_clock::now();

    // -------- SETUP --------
    OpenCLContext ctx;
    opencl_setup(ctx, numbers, device_string);

    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time_seconds);

    std::uint64_t iters = 0;
    float calculated_value{};

    // -------- COMPUTE LOOP --------
    do {
        calculated_value = parallel_task(ctx, N);
        iters++;
    } while (std::chrono::steady_clock::now() < deadline);

    auto t2 = std::chrono::steady_clock::now();

    // -------- CLEANUP --------
    opencl_cleanup(ctx);

    auto t3 = std::chrono::steady_clock::now();

    // Results
    //==================================================

    auto time_setup = std::chrono::duration<double>(t1 - t0).count();
    auto time_calc = std::chrono::duration<double>(t2 - t1).count();
    auto time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    auto time_total = std::chrono::duration<double>(t3 - t0).count();
    auto time_per_iteration = time_calc / iters;

    {
        const auto method{std::string("Parallel OpenCL 32")};
        const auto operation_string = std::string("sum");
        const auto comments{std::string("operation:") + operation_string};

        const std::string base_file_name = "results/parallel_opencl_32";
        const std::string json_file = base_file_name + "_" + helper::random_suffix(12) + ".json";

        nlohmann::json j;

        j["file"] = json_file;
        j["method"] = method;
        j["operation"] = operation_string;
        j["comments"] = comments;
        j["threads"] = 0;
        j["precision"] = "64";
        j["device"] = "OpenCL";

        j["test_time_seconds"] = test_time_seconds;
        j["iterations"] = iters;
        j["time_per_iteration"] = time_per_iteration;
        j["time_setup"] = time_setup;
        j["time_calc"] = time_calc;
        j["time_cleanup"] = time_cleanup;
        j["time_total"] = time_total;

        j["calculated_value"] = helper::to_string_precise(calculated_value);
        j["values"] = helper::to_string_precise_vector(numbers);

        j["max_rss_kb"] = helper::max_rss_kb();

        std::ofstream out(json_file);
        if (!out)
            throw std::runtime_error("Failed to open output JSON file.");

        out << std::setw(2) << j << '\n';
    }

    return 0;
}
