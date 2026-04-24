/**
 * @file parallel_opencl_32.cpp
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

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "helper/Error.hpp"
#include "helper/helper.hpp"

#include <nlohmann/json.hpp>

using OperationKind = helper::OperationKind;

namespace {

inline constexpr float MIN_DENOMINATOR{1.0e-9};
inline constexpr std::uint64_t RNG_SEED{123456789ULL};

void opencl_check(cl_int status, const char* message) {
    if (status != CL_SUCCESS) {
        std::ostringstream oss;
        oss << message << " (OpenCL error " << status << ")";
        THROW_RUNTIME_ERROR(oss.str());
    }
}

[[nodiscard]]
const char* kernel_source() {
    return R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel_add(
    const ulong n,
    __global const float* a,
    __global const float* b,
    __global float* c)
{
    const ulong idx = (ulong)get_global_id(0);
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__kernel void kernel_multiply(
    const ulong n,
    __global const float* a,
    __global const float* b,
    __global float* c)
{
    const ulong idx = (ulong)get_global_id(0);
    if (idx < n)
    {
        c[idx] = a[idx] * b[idx];
    }
}

__kernel void kernel_divide(
    const ulong n,
    __global const float* a,
    __global const float* b,
    __global float* c)
{
    const ulong idx = (ulong)get_global_id(0);
    if (idx < n)
    {
        const float denom = b[idx] > 1.0e-9 ? b[idx] : 1.0e-9;
        c[idx] = a[idx] / denom;
    }
}

__kernel void kernel_power(
    const ulong n,
    __global const float* a,
    __global const float* b,
    __global float* c)
{
    const ulong idx = (ulong)get_global_id(0);
    if (idx < n)
    {
        c[idx] = pow(a[idx], b[idx]);
    }
}

__kernel void kernel_exp(
    const ulong n,
    __global const float* a,
    __global const float* b,
    __global float* c)
{
    const ulong idx = (ulong)get_global_id(0);
    if (idx < n)
    {
        c[idx] = exp(a[idx]) + exp(b[idx]);
    }
}

__kernel void kernel_log(
    const ulong n,
    __global const float* a,
    __global const float* b,
    __global float* c)
{
    const ulong idx = (ulong)get_global_id(0);
    if (idx < n)
    {
        c[idx] = log(a[idx]) + log(b[idx]);
    }
}

__kernel void kernel_sqrt(
    const ulong n,
    __global const float* a,
    __global const float* b,
    __global float* c)
{
    const ulong idx = (ulong)get_global_id(0);
    if (idx < n)
    {
        c[idx] = sqrt(a[idx]) + sqrt(b[idx]);
    }
}
)CLC";
}

[[nodiscard]]
const char* kernel_name(OperationKind operation) {
    switch (operation) {
        case OperationKind::Add: {
            return "kernel_add";
        }
        case OperationKind::Multiply: {
            return "kernel_multiply";
        }
        case OperationKind::Divide: {
            return "kernel_divide";
        }
        case OperationKind::Power: {
            return "kernel_power";
        }
        case OperationKind::Exp: {
            return "kernel_exp";
        }
        case OperationKind::Log: {
            return "kernel_log";
        }
        case OperationKind::Sqrt: {
            return "kernel_sqrt";
        }
    }

    THROW_RUNTIME_ERROR("Unhandled OperationKind value.");
}

[[nodiscard]]
std::string get_platform_string(cl_platform_id platform, cl_platform_info param) {
    std::size_t size{0};
    opencl_check(clGetPlatformInfo(platform, param, 0, nullptr, &size),
                 "clGetPlatformInfo(size) failed.");

    std::string value(size, '\0');
    opencl_check(clGetPlatformInfo(platform, param, size, value.data(), nullptr),
                 "clGetPlatformInfo(data) failed.");

    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

[[nodiscard]]
std::string get_device_string(cl_device_id device, cl_device_info param) {
    std::size_t size{0};
    opencl_check(clGetDeviceInfo(device, param, 0, nullptr, &size),
                 "clGetDeviceInfo(size) failed.");

    std::string value(size, '\0');
    opencl_check(clGetDeviceInfo(device, param, size, value.data(), nullptr),
                 "clGetDeviceInfo(data) failed.");

    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

[[nodiscard]]
cl_device_id pick_device(std::string_view device_string) {
    cl_uint platform_count{0};
    const cl_int status = clGetPlatformIDs(0, nullptr, &platform_count);

    if (status == CL_PLATFORM_NOT_FOUND_KHR) {
        THROW_RUNTIME_ERROR(
            "No OpenCL platform found. "
            "Install an OpenCL ICD/runtime such as pocl-opencl-icd, "
            "or enable vendor OpenCL support inside the container.");
    }

    opencl_check(status, "clGetPlatformIDs(count) failed.");

    if (platform_count == 0) {
        THROW_RUNTIME_ERROR("No OpenCL platforms found.");
    }

    auto platforms = std::vector<cl_platform_id>(platform_count);
    opencl_check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
                 "clGetPlatformIDs(data) failed.");

    const cl_device_type requested_type =
        (device_string == "CPU") ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;

    for (const auto platform : platforms) {
        cl_uint device_count{0};
        const auto device_status =
            clGetDeviceIDs(platform, requested_type, 0, nullptr, &device_count);
        if (device_status == CL_SUCCESS && device_count > 0) {
            auto devices = std::vector<cl_device_id>(device_count);
            opencl_check(
                clGetDeviceIDs(platform, requested_type, device_count, devices.data(), nullptr),
                "clGetDeviceIDs(requested type) failed.");
            return devices.front();
        }
    }

    std::ostringstream oss;
    oss << "No suitable OpenCL " << device_string << " device found.";
    THROW_RUNTIME_ERROR(oss.str());
}

void print_build_log(cl_program program, cl_device_id device) {
    std::size_t log_size{0};
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

    if (log_size > 1) {
        std::string build_log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(),
                              nullptr);
        std::cerr << "OpenCL build log:\n" << build_log << "\n";
    }
}

}  // namespace

/**
 * @brief Entry point into program.
 */
auto main(int argc, char** argv) -> int {
    cl_context context{nullptr};
    cl_command_queue queue{nullptr};
    cl_program program{nullptr};
    cl_kernel kernel{nullptr};
    cl_mem dev_a{nullptr};
    cl_mem dev_b{nullptr};
    cl_mem dev_c{nullptr};

    try {
        if (argc < 4) {
            THROW_INVALID_ARGUMENT("Usage: serial.x time_limit vec_size operation");
        }

        const auto test_time_seconds{helper::parse_floating_point(argv[1])};
        const auto n{helper::parse_size(argv[2])};
        const auto operation_string{std::string_view(argv[3])};
        const auto operation{helper::parse_operation(operation_string)};

        std::string_view device_string = "GPU";
        if (argc >= 5) {
            device_string = argv[4];

            if (device_string != "GPU" && device_string != "CPU") {
                THROW_INVALID_ARGUMENT("device must be GPU or CPU");
            }
        }

        if (test_time_seconds <= 0.0) {
            THROW_INVALID_ARGUMENT("time_limit must be > 0.");
        }
        if (n == 0) {
            THROW_INVALID_ARGUMENT("vec_size must be > 0.");
        }

        std::mt19937_64 rng(RNG_SEED);
        std::uniform_real_distribution<double> dist(1.0, 2.0);

        auto numbers_a{std::vector<float>{}};
        auto numbers_b{std::vector<float>{}};
        numbers_a.reserve(n);
        numbers_b.reserve(n);

        for (auto i = std::size_t(0); i < n; i++) {
            numbers_a.emplace_back(dist(rng));
            numbers_b.emplace_back(dist(rng));
        }

        // ======= Calculation Starts ========
        const auto t0{std::chrono::steady_clock::now()};

        const auto device{pick_device(device_string)};

        const auto device_name{get_device_string(device, CL_DEVICE_NAME)};
        std::cerr << "Using device: " << device_name << "\n";

        // Check FP64 support
        cl_device_fp_config fp64_config{};
        opencl_check(clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64_config),
                                     &fp64_config, nullptr),
                     "clGetDeviceInfo(CL_DEVICE_DOUBLE_FP_CONFIG) failed.");

        if (fp64_config == 0) {
            THROW_RUNTIME_ERROR("Selected OpenCL device does not support double precision.");
        }

        cl_int status{CL_SUCCESS};
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        opencl_check(status, "clCreateContext failed.");

        queue = clCreateCommandQueue(context, device, 0, &status);
        opencl_check(status, "clCreateCommandQueue failed.");

        const char* source{kernel_source()};
        const std::size_t source_length{std::char_traits<char>::length(source)};
        program = clCreateProgramWithSource(context, 1, &source, &source_length, &status);
        opencl_check(status, "clCreateProgramWithSource failed.");

        status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            print_build_log(program, device);
            opencl_check(status, "clBuildProgram failed.");
        }

        kernel = clCreateKernel(program, kernel_name(operation), &status);
        opencl_check(status, "clCreateKernel failed.");

        auto numbers_c{std::vector<float>(n)};

        dev_a = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &status);
        opencl_check(status, "clCreateBuffer(dev_a) failed.");

        dev_b = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &status);
        opencl_check(status, "clCreateBuffer(dev_b) failed.");

        dev_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &status);
        opencl_check(status, "clCreateBuffer(dev_c) failed.");

        opencl_check(clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, n * sizeof(float),
                                          numbers_a.data(), 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(dev_a) failed.");
        opencl_check(clEnqueueWriteBuffer(queue, dev_b, CL_TRUE, 0, n * sizeof(float),
                                          numbers_b.data(), 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(dev_b) failed.");

        const auto n_opencl{static_cast<cl_ulong>(n)};
        opencl_check(clSetKernelArg(kernel, 0, sizeof(cl_ulong), &n_opencl),
                     "clSetKernelArg(0) failed.");
        opencl_check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev_a),
                     "clSetKernelArg(1) failed.");
        opencl_check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev_b),
                     "clSetKernelArg(2) failed.");
        opencl_check(clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev_c),
                     "clSetKernelArg(3) failed.");

        // ======= Start up =======
        const auto t1{std::chrono::steady_clock::now()};
        const auto deadline{t1 + std::chrono::duration<double>(test_time_seconds)};

        constexpr std::size_t local_size{256};
        const std::size_t global_size{((n + local_size - 1) / local_size) * local_size};

        auto iters{static_cast<std::uint64_t>(0)};

        do {
            opencl_check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size,
                                                &local_size, 0, nullptr, nullptr),
                         "clEnqueueNDRangeKernel failed.");

            opencl_check(clFinish(queue), "clFinish inside timed loop failed.");

            iters++;
        } while (std::chrono::steady_clock::now() < deadline);

        opencl_check(clEnqueueReadBuffer(queue, dev_c, CL_TRUE, 0, n * sizeof(float),
                                         numbers_c.data(), 0, nullptr, nullptr),
                     "clEnqueueReadBuffer(dev_c) failed.");

        // ======= Clean up =======
        const auto t2{std::chrono::steady_clock::now()};

        // Free device allocations
        opencl_check(clReleaseMemObject(dev_a), "clReleaseMemObject(dev_a) failed.");
        dev_a = nullptr;

        opencl_check(clReleaseMemObject(dev_b), "clReleaseMemObject(dev_b) failed.");
        dev_b = nullptr;

        opencl_check(clReleaseMemObject(dev_c), "clReleaseMemObject(dev_c) failed.");
        dev_c = nullptr;

        opencl_check(clReleaseKernel(kernel), "clReleaseKernel failed.");
        kernel = nullptr;

        opencl_check(clReleaseProgram(program), "clReleaseProgram failed.");
        program = nullptr;

        opencl_check(clReleaseCommandQueue(queue), "clReleaseCommandQueue failed.");
        queue = nullptr;

        opencl_check(clReleaseContext(context), "clReleaseContext failed.");
        context = nullptr;

        // ======= Calculation Ends ========

        const auto t3{std::chrono::steady_clock::now()};

        const auto calculated_value{helper::check_sum(numbers_c)};

        const auto time_setup{std::chrono::duration<double>(t1 - t0).count()};
        const auto time_calc{std::chrono::duration<double>(t2 - t1).count()};
        const auto time_cleanup{std::chrono::duration<double>(t3 - t2).count()};
        const auto time_total{std::chrono::duration<double>(t3 - t0).count()};
        const auto time_per_iteration{time_calc / static_cast<double>(iters)};

        const auto method{std::string("Parallel OpenCL 32 ") + std::string(device_string)};
        const auto comments{std::string("operation:") + std::string(operation_string)};

        // Output
        {
            const std::string base_file_name =
                "results/parallel_opencl_32_" + std::string(operation_string);
            const std::string json_file =
                base_file_name + "_" + helper::random_suffix(12) + ".json";

            nlohmann::json j;

            // Metadata / identity
            j["file"] = json_file;
            j["method"] = method;
            j["operation"] = operation_string;
            j["comments"] = comments;
            j["threads"] = 1;
            j["device"] = std::string(device_string);
            j["precision"] = "32";
            j["device"] = device_string;

            // Iteration/timing
            j["test_time_seconds"] = test_time_seconds;
            j["iterations"] = iters;
            j["time_per_iteration"] = time_per_iteration;
            j["time_setup"] = time_setup;
            j["time_calc"] = time_calc;
            j["time_cleanup"] = time_cleanup;
            j["time_total"] = time_total;

            // Values
            j["calculated_value"] = helper::to_string_precise(calculated_value);
            j["values"] = helper::to_string_precise_vector(numbers_c);

            // Memory
            // j["max_rss_kb"] = max_rss_kb();

            std::ofstream out(json_file);
            if (!out) {
                throw std::runtime_error("Failed to open output JSON file.");
            }

            // Save JSON file.
            out << std::setw(2) << j << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
