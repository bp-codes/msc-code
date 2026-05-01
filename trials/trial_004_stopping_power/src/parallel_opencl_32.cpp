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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "helper/Error.hpp"
#include "helper/helper.hpp"

#include <nlohmann/json.hpp>

#include <CL/cl.h>


static inline void check_opencl_error(const cl_int err, const char* const message) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error: " << err << " : " << message << '\n';
        std::exit(2);
    }
}

/**
 * @brief Host log function.
 */
[[nodiscard]]
static inline float host_compatible_log(const float x) {
    return std::log(x);
}

/**
 * @brief Host sqrt function.
 */
[[nodiscard]]
static inline float host_compatible_sqrt(const float x) {
    return std::sqrt(x);
}

/**
 * @brief Linear stopping power (dE/dx) for a charged ion in a material using the PDG Bethe
 * equation.
 */
[[nodiscard]]
static inline float stopping_power(
    const float projectile_velocity_ms, const int projectile_atomic_number,
    const float projectile_atomic_mass_mev, const int target_atomic_number,
    const float target_atomic_mass_g_mol, const float target_density_g_cm3,
    const float mean_excitation_energy_mev, const float density_effect_delta) {
    static constexpr auto SPEED_OF_LIGHT_MS{299792458.0f};
    static constexpr auto ELECTRON_MASS_MEV{0.51099895000f};
    static constexpr auto BETHE_CONSTANT_K{0.307075f};
    static constexpr auto SMALL_VALUE{1.0e-9f};

    const auto beta_raw{projectile_velocity_ms / SPEED_OF_LIGHT_MS};
    const auto beta{std::clamp(beta_raw, SMALL_VALUE, 0.99999f)};
    const auto beta2{beta * beta};

    const auto inv_one_minus_beta2{1.0f / (1.0f - beta2)};
    const auto gamma2{std::max(0.0f, inv_one_minus_beta2)};
    const auto gamma{host_compatible_sqrt(gamma2)};

    const auto total_energy_mev{std::max(0.0f, gamma * projectile_atomic_mass_mev)};
    (void)total_energy_mev;

    const auto electron_to_projectile_mass{ELECTRON_MASS_MEV /
                                           std::max(SMALL_VALUE, projectile_atomic_mass_mev)};

    const auto w_max_numerator{2.0f * ELECTRON_MASS_MEV * beta2 * gamma2};

    const auto w_max_denominator =
        std::max(1.0f + 2.0f * gamma * electron_to_projectile_mass +
                     (electron_to_projectile_mass * electron_to_projectile_mass),
                 SMALL_VALUE);

    const auto w_max_mev{w_max_numerator / w_max_denominator};

    const auto mean_excitation_energy2_mev2{mean_excitation_energy_mev *
                                            mean_excitation_energy_mev};

    const auto log_argument = std::max((2.0f * ELECTRON_MASS_MEV * beta2 * gamma2 * w_max_mev) /
                                           std::max(SMALL_VALUE, mean_excitation_energy2_mev2),
                                       SMALL_VALUE);

    const auto bracket =
        0.5f * host_compatible_log(log_argument) - beta2 - 0.5f * density_effect_delta;

    const auto projectile_charge{static_cast<float>(projectile_atomic_number)};
    const auto projectile_charge2{projectile_charge * projectile_charge};

    const auto z_over_a{static_cast<float>(target_atomic_number) /
                        std::max(SMALL_VALUE, target_atomic_mass_g_mol)};

    const auto prefactor_mass{BETHE_CONSTANT_K * projectile_charge2 * z_over_a / beta2};

    const auto mass_stopping_power_mev_cm2_per_g{prefactor_mass * bracket};
    const auto linear_stopping_power_mev_per_cm{target_density_g_cm3 *
                                                mass_stopping_power_mev_cm2_per_g};

    return linear_stopping_power_mev_per_cm;
}

// Parallel versions

void opencl_check(cl_int status, const char* message) {
    if (status != CL_SUCCESS) {
        std::ostringstream oss;
        oss << message << " (OpenCL error " << status << ")";
        THROW_RUNTIME_ERROR(oss.str());
    }
}

[[nodiscard]]
static inline const char* kernel_source() {
    return R"CLC(
inline float clamp_float(const float x, const float lo, const float hi)
{
    return fmin(fmax(x, lo), hi);
}

inline float stopping_power_kernel(
    const float projectile_velocity_ms,
    const int projectile_atomic_number,
    const float projectile_atomic_mass_mev,
    const int target_atomic_number,
    const float target_atomic_mass_g_mol,
    const float target_density_g_cm3,
    const float mean_excitation_energy_mev,
    const float density_effect_delta)
{
    const float SPEED_OF_LIGHT_MS = 299792458.0f;
    const float ELECTRON_MASS_MEV = 0.51099895000f;
    const float BETHE_CONSTANT_K  = 0.307075f;
    const float SMALL_VALUE       = 1.0e-9f;

    const float beta_raw = projectile_velocity_ms / SPEED_OF_LIGHT_MS;
    const float beta = clamp_float(beta_raw, SMALL_VALUE, 0.99999f);
    const float beta2 = beta * beta;

    const float inv_one_minus_beta2 = 1.0f / (1.0f - beta2);
    const float gamma2 = fmax(0.0f, inv_one_minus_beta2);
    const float gamma = sqrt(gamma2);

    const float electron_to_projectile_mass =
        ELECTRON_MASS_MEV / fmax(SMALL_VALUE, projectile_atomic_mass_mev);

    const float w_max_numerator = 2.0f * ELECTRON_MASS_MEV * beta2 * gamma2;

    const float w_max_denominator = fmax(
        1.0f
      + 2.0f * gamma * electron_to_projectile_mass
      + (electron_to_projectile_mass * electron_to_projectile_mass),
        SMALL_VALUE);

    const float w_max_mev = w_max_numerator / w_max_denominator;

    const float mean_excitation_energy2_mev2 =
        mean_excitation_energy_mev * mean_excitation_energy_mev;

    const float log_argument = fmax(
        (2.0f * ELECTRON_MASS_MEV * beta2 * gamma2 * w_max_mev)
      / fmax(SMALL_VALUE, mean_excitation_energy2_mev2),
        SMALL_VALUE);

    const float bracket =
        0.5f * log(log_argument)
      - beta2
      - 0.5f * density_effect_delta;

    const float projectile_charge = (float)projectile_atomic_number;
    const float projectile_charge2 = projectile_charge * projectile_charge;

    const float z_over_a =
        (float)target_atomic_number / fmax(SMALL_VALUE, target_atomic_mass_g_mol);

    const float prefactor_mass =
        BETHE_CONSTANT_K * projectile_charge2 * z_over_a / beta2;

    const float mass_stopping_power_mev_cm2_per_g = prefactor_mass * bracket;
    const float linear_stopping_power_mev_per_cm =
        target_density_g_cm3 * mass_stopping_power_mev_cm2_per_g;

    return linear_stopping_power_mev_per_cm;
}

__kernel void stopping_power_kernel_main(
    __global const float* velocity_device,
    __global float* stopping_power_values_device,
    const ulong n)
{
    const size_t i = get_global_id(0);

    if (i >= n)
    {
        return;
    }

    const int PROJECTILE_ATOMIC_NUMBER = 1;
    const float PROJECTILE_ATOMIC_MASS_MEV = 938.2720813f;

    const int TARGET_ATOMIC_NUMBER = 26;
    const float TARGET_ATOMIC_MASS_G_MOL = 55.845f;
    const float TARGET_DENSITY_G_CM3 = 7.874f;

    const float MEAN_EXCITATION_ENERGY_MEV = 286.0e-6f;
    const float DENSITY_EFFECT_DELTA = 0.0f;

    stopping_power_values_device[i] = stopping_power_kernel(
        velocity_device[i],
        PROJECTILE_ATOMIC_NUMBER,
        PROJECTILE_ATOMIC_MASS_MEV,
        TARGET_ATOMIC_NUMBER,
        TARGET_ATOMIC_MASS_G_MOL,
        TARGET_DENSITY_G_CM3,
        MEAN_EXCITATION_ENERGY_MEV,
        DENSITY_EFFECT_DELTA);
}
)CLC";
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

auto main(int argc, char** argv) -> int {
    try {
        if (argc < 3) {
            THROW_INVALID_ARGUMENT("Usage: serial.x time_limit vec_size operation");
        }

        const auto test_time_seconds{helper::parse_floating_point(argv[1])};
        const auto n{helper::parse_size(argv[2])};

        if (n <= 0) {
            std::cerr << "vec_size must be a positive integer\n";
            return 1;
        }

        std::string_view device_string = "GPU";
        if (argc >= 4) {
            device_string = argv[3];

            if (device_string != "GPU" && device_string != "CPU") {
                THROW_INVALID_ARGUMENT("device must be GPU or CPU");
            }
        }

        // Random number generator
        std::mt19937_64 rng(123456789ULL);
        std::uniform_real_distribution<double> dist(1.0e7, 1.0e8);  // [0.0, 1.0)

        // Vector of numbers
        auto velocity_array{std::vector<float>{}};
        velocity_array.reserve(n);

        // Populate vectors
        std::generate_n(std::back_inserter(velocity_array), n,
                        [&]() { return static_cast<float>(dist(rng)); });

        // ======= Set up before calculation ========
        const auto t0{std::chrono::steady_clock::now()};

        // Vector to store stopping power values
        auto stopping_power_values{std::vector<float>(n)};

        cl_context context{nullptr};
        cl_command_queue queue{nullptr};
        cl_program program{nullptr};
        cl_kernel kernel{nullptr};
        cl_mem velocity_array_device{nullptr};
        cl_mem stopping_power_values_device{nullptr};

        const auto device{pick_device(device_string)};

        const auto device_name{get_device_string(device, CL_DEVICE_NAME)};
        std::cerr << "Using device: " << device_name << "\n";

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

        // Copy velocity vector to device
        kernel = clCreateKernel(program, "stopping_power_kernel_main", &status);
        opencl_check(status, "clCreateKernel failed.");

        velocity_array_device =
            clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &status);
        opencl_check(status, "clCreateBuffer(velocity_array_device) failed.");

        stopping_power_values_device =
            clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &status);
        opencl_check(status, "clCreateBuffer(stopping_power_values_device) failed.");

        opencl_check(
            clEnqueueWriteBuffer(queue, velocity_array_device, CL_TRUE, 0, n * sizeof(float),
                                 velocity_array.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(velocity_array_device) failed.");

        const auto n_opencl{static_cast<cl_ulong>(n)};

        opencl_check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &velocity_array_device),
                     "clSetKernelArg(0) failed.");

        opencl_check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &stopping_power_values_device),
                     "clSetKernelArg(1) failed.");

        opencl_check(clSetKernelArg(kernel, 2, sizeof(cl_ulong), &n_opencl),
                     "clSetKernelArg(2) failed.");

        // ======= Carry out calculation ========
        const auto t1{std::chrono::steady_clock::now()};
        const auto deadline{t1 + std::chrono::duration<double>(test_time_seconds)};

        constexpr std::size_t local_size{256};
        const std::size_t global_size{((n + local_size - 1) / local_size) * local_size};

        auto iters{std::uint64_t(0)};

        do {
            opencl_check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size,
                                                &local_size, 0, nullptr, nullptr),
                         "clEnqueueNDRangeKernel failed.");

            opencl_check(clFinish(queue), "clFinish inside timed loop failed.");

            iters++;
        } while (std::chrono::steady_clock::now() < deadline);

        // ======= Copy back and clean up after calculation ========
        const auto t2{std::chrono::steady_clock::now()};

        opencl_check(
            clEnqueueReadBuffer(queue, stopping_power_values_device, CL_TRUE, 0, n * sizeof(float),
                                stopping_power_values.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(dev_c) failed.");

        const auto calculated_value{static_cast<double>(helper::check_sum(stopping_power_values))};

        // Free device allocations
        opencl_check(clReleaseMemObject(stopping_power_values_device),
                     "clReleaseMemObject(dev_a) failed.");
        opencl_check(clReleaseMemObject(velocity_array_device),
                     "clReleaseMemObject(dev_b) failed.");
        opencl_check(clReleaseKernel(kernel), "clReleaseKernel failed.");
        opencl_check(clReleaseProgram(program), "clReleaseProgram failed.");
        opencl_check(clReleaseCommandQueue(queue), "clReleaseCommandQueue failed.");
        opencl_check(clReleaseContext(context), "clReleaseContext failed.");
        stopping_power_values_device = nullptr;
        velocity_array_device = nullptr;
        kernel = nullptr;
        program = nullptr;
        queue = nullptr;
        context = nullptr;

        // ======= End ========
        const auto t3{std::chrono::steady_clock::now()};

        const auto time_setup_s{std::chrono::duration<double>(t1 - t0).count()};
        const auto time_calc_s{std::chrono::duration<double>(t2 - t1).count()};
        const auto time_cleanup_s{std::chrono::duration<double>(t3 - t2).count()};
        const auto time_total_s{std::chrono::duration<double>(t3 - t0).count()};
        const auto time_per_iteration_s{(iters > 0) ? (time_calc_s / static_cast<double>(iters))
                                                    : 0.0};

        const auto method{std::string("Parallel OpenCL 32")};
        const auto comments{std::string("stopping_power")};

        {
            const std::string base_file_name = "results/parallel_opencl_32";
            const std::string json_file =
                base_file_name + "_" + helper::random_suffix(12) + ".json";

            // Cast to double for output
            auto stopping_power_values_out{std::vector<double>{}};
            stopping_power_values_out.reserve(stopping_power_values.size());
            for (auto i = std::size_t(0); i < stopping_power_values.size(); i++) {
                stopping_power_values_out.emplace_back(
                    static_cast<double>(stopping_power_values[i]));
            }

            nlohmann::json j;

            j["file"] = json_file;
            j["method"] = method;
            j["operation"] = "Bethe-Bloch Stopping Power";
            j["comments"] = comments;
            j["threads"] = 1;
            j["device"] = std::string(device_string);

            j["test_time_seconds"] = test_time_seconds;
            j["iterations"] = iters;
            j["time_per_iteration"] = time_per_iteration_s;
            j["time_setup"] = time_setup_s;
            j["time_calc"] = time_calc_s;
            j["time_cleanup"] = time_cleanup_s;
            j["time_total"] = time_total_s;

            j["calculated_value"] = helper::to_string_precise(calculated_value);
            j["values"] = helper::to_string_precise_vector(stopping_power_values_out);

            std::ofstream out(json_file);
            if (!out) {
                throw std::runtime_error("Failed to open output JSON file.");
            }

            out << std::setw(2) << j << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
