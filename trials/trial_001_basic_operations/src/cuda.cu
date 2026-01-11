// sycl.cu
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <cuda_runtime.h>

#include "Error.hpp"

namespace
{

inline constexpr double MIN_DENOMINATOR {1.0e-9};
inline constexpr std::uint64_t RNG_SEED {123456789ULL};



[[nodiscard]]
std::string random_suffix(const std::size_t n)
{
    static constexpr char charset[] =
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789";

    static thread_local std::mt19937 rng{RNG_SEED};
    std::uniform_int_distribution<std::size_t> dist(0, sizeof(charset) - 2);

    std::string s;
    s.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        s.push_back(charset[dist(rng)]);
    }
    return s;
}



/**
 * @enum OperationKind
 * @brief Supported element-wise operations.
 */
enum class OperationKind
{
    Add,
    Multiply,
    Divide,
    Power,
    Exp,
    Log,
    Sqrt
};



/**
 * @brief Parse an operation string into an OperationKind.
 * @param operation Operation name (e.g. "add")
 * @return Parsed enum value.
 * @throws std::invalid_argument if the operation is unknown.
 */
[[nodiscard]]
OperationKind parse_operation(std::string_view operation)
{
    if (operation == "add")
    {
        return OperationKind::Add;
    }
    if (operation == "multiply")
    {
        return OperationKind::Multiply;
    }
    if (operation == "divide")
    {
        return OperationKind::Divide;
    }
    if (operation == "power")
    {
        return OperationKind::Power;
    }
    if (operation == "exp")
    {
        return OperationKind::Exp;
    }
    if (operation == "log")
    {
        return OperationKind::Log;
    }
    if (operation == "sqrt")
    {
        return OperationKind::Sqrt;
    }

    THROW_INVALID_ARGUMENT("Unknown operation.");
}



/**
 * @brief Parse a double from argv using std::from_chars.
 * @param s Null-terminated string.
 * @return Parsed double.
 * @throws std::invalid_argument on parse failure.
 */
[[nodiscard]]
double parse_double(const char* s)
{
    if (s == nullptr)
    {
        THROW_INVALID_ARGUMENT("Null argument encountered while parsing double.");
    }

    double value {};
    const auto* first {s};
    const auto* last {s + std::char_traits<char>::length(s)};

    const auto result {std::from_chars(first, last, value)};
    if (result.ec != std::errc{} || result.ptr != last)
    {
        THROW_INVALID_ARGUMENT("Failed to parse double argument.");
    }

    return value;
}



/**
 * @brief Parse a non-negative std::size_t from argv using std::from_chars.
 * @param s Null-terminated string.
 * @return Parsed size.
 * @throws std::invalid_argument on parse failure.
 */
[[nodiscard]]
std::size_t parse_size(const char* s)
{
    if (s == nullptr)
    {
        THROW_INVALID_ARGUMENT("Null argument encountered while parsing size.");
    }

    std::size_t value {};
    const auto* first {s};
    const auto* last {s + std::char_traits<char>::length(s)};

    const auto result {std::from_chars(first, last, value)};
    if (result.ec != std::errc{} || result.ptr != last)
    {
        THROW_INVALID_ARGUMENT("Failed to parse size argument.");
    }

    return value;
}



/**
 * @brief Validate that inputs/outputs are consistent before entering compute loops.
 * @param numbers_a First input vector.
 * @param numbers_b Second input vector.
 * @param numbers_c Output vector (must be pre-sized).
 * @throws std::invalid_argument if sizes do not match.
 */
template<typename T>
void validate_sizes(
    const std::vector<T>& numbers_a,
    const std::vector<T>& numbers_b,
    const std::vector<T>& numbers_c)
{
    if (numbers_a.size() != numbers_b.size())
    {
        THROW_INVALID_ARGUMENT("Input vectors must have the same length.");
    }
    if (numbers_c.size() != numbers_a.size())
    {
        THROW_INVALID_ARGUMENT("Output vector must be pre-sized to match inputs.");
    }
}



// Serial versions (unchanged)



/**
 * @brief Element-wise addition: c[i] = a[i] + b[i]
 */
void serial_add(
    const std::vector<double>& numbers_a,
    const std::vector<double>& numbers_b,
    std::vector<double>& numbers_c)
{
    const auto n {std::size_t(numbers_a.size())};
    for (auto i = std::size_t(0); i < n; i++)
    {
        numbers_c[i] = numbers_a[i] + numbers_b[i];
    }
}



/**
 * @brief Element-wise multiplication: c[i] = a[i] * b[i]
 */
void serial_multiply(
    const std::vector<double>& numbers_a,
    const std::vector<double>& numbers_b,
    std::vector<double>& numbers_c)
{
    const auto n {std::size_t(numbers_a.size())};
    for (auto i = std::size_t(0); i < n; i++)
    {
        numbers_c[i] = numbers_a[i] * numbers_b[i];
    }
}



/**
 * @brief Element-wise division: c[i] = a[i] / max(b[i], MIN_DENOMINATOR)
 */
void serial_divide(
    const std::vector<double>& numbers_a,
    const std::vector<double>& numbers_b,
    std::vector<double>& numbers_c)
{
    const auto n {std::size_t(numbers_a.size())};
    for (auto i = std::size_t(0); i < n; i++)
    {
        numbers_c[i] = numbers_a[i] / std::max(numbers_b[i], MIN_DENOMINATOR);
    }
}



/**
 * @brief Element-wise power: c[i] = pow(a[i], b[i])
 */
void serial_power(
    const std::vector<double>& numbers_a,
    const std::vector<double>& numbers_b,
    std::vector<double>& numbers_c)
{
    const auto n {std::size_t(numbers_a.size())};
    for (auto i = std::size_t(0); i < n; i++)
    {
        numbers_c[i] = std::pow(numbers_a[i], numbers_b[i]);
    }
}



/**
 * @brief Element-wise exp sum: c[i] = exp(a[i]) + exp(b[i])
 */
void serial_exp(
    const std::vector<double>& numbers_a,
    const std::vector<double>& numbers_b,
    std::vector<double>& numbers_c)
{
    const auto n {std::size_t(numbers_a.size())};
    for (auto i = std::size_t(0); i < n; i++)
    {
        numbers_c[i] = std::exp(numbers_a[i]) + std::exp(numbers_b[i]);
    }
}



/**
 * @brief Element-wise log sum: c[i] = log(a[i]) + log(b[i])
 * @warning Inputs must be > 0. No bounds/validity checking is performed in this hot loop.
 */
void serial_log(
    const std::vector<double>& numbers_a,
    const std::vector<double>& numbers_b,
    std::vector<double>& numbers_c)
{
    const auto n {std::size_t(numbers_a.size())};
    for (auto i = std::size_t(0); i < n; i++)
    {
        numbers_c[i] = std::log(numbers_a[i]) + std::log(numbers_b[i]);
    }
}



/**
 * @brief Element-wise sqrt sum: c[i] = sqrt(a[i]) + sqrt(b[i])
 * @warning Inputs must be >= 0. No bounds/validity checking is performed in this hot loop.
 */
void serial_sqrt(
    const std::vector<double>& numbers_a,
    const std::vector<double>& numbers_b,
    std::vector<double>& numbers_c)
{
    const auto n {std::size_t(numbers_a.size())};
    for (auto i = std::size_t(0); i < n; i++)
    {
        numbers_c[i] = std::sqrt(numbers_a[i]) + std::sqrt(numbers_b[i]);
    }
}



/**
 * @brief Dispatch the selected operation.
 * @param operation Operation kind.
 * @param numbers_a First input vector.
 * @param numbers_b Second input vector.
 * @param numbers_c Output vector (must be pre-sized).
 */
void serial_task(
    OperationKind operation,
    const std::vector<double>& numbers_a,
    const std::vector<double>& numbers_b,
    std::vector<double>& numbers_c)
{
    switch (operation)
    {
        case OperationKind::Add:
        {
            serial_add(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Multiply:
        {
            serial_multiply(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Divide:
        {
            serial_divide(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Power:
        {
            serial_power(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Exp:
        {
            serial_exp(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Log:
        {
            serial_log(numbers_a, numbers_b, numbers_c);
            return;
        }
        case OperationKind::Sqrt:
        {
            serial_sqrt(numbers_a, numbers_b, numbers_c);
            return;
        }
    }

    THROW_RUNTIME_ERROR("Unhandled OperationKind value.");
}



// CUDA parallel versions



inline void cuda_check(cudaError_t status, const char* message)
{
    if (status != cudaSuccess)
    {
        (void)status;
        THROW_RUNTIME_ERROR(message);
    }
}



__global__ void kernel_add(std::size_t n, const double* a, const double* b, double* c)
{
    const auto idx {std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void kernel_multiply(std::size_t n, const double* a, const double* b, double* c)
{
    const auto idx {std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n)
    {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void kernel_divide(std::size_t n, const double* a, const double* b, double* c)
{
    const auto idx {std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n)
    {
        const auto denom {b[idx] > MIN_DENOMINATOR ? b[idx] : MIN_DENOMINATOR};
        c[idx] = a[idx] / denom;
    }
}

__global__ void kernel_power(std::size_t n, const double* a, const double* b, double* c)
{
    const auto idx {std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n)
    {
        c[idx] = pow(a[idx], b[idx]);
    }
}

__global__ void kernel_exp(std::size_t n, const double* a, const double* b, double* c)
{
    const auto idx {std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n)
    {
        c[idx] = exp(a[idx]) + exp(b[idx]);
    }
}

__global__ void kernel_log(std::size_t n, const double* a, const double* b, double* c)
{
    const auto idx {std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n)
    {
        c[idx] = log(a[idx]) + log(b[idx]);
    }
}

__global__ void kernel_sqrt(std::size_t n, const double* a, const double* b, double* c)
{
    const auto idx {std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x)};
    if (idx < n)
    {
        c[idx] = sqrt(a[idx]) + sqrt(b[idx]);
    }
}



cudaError_t launch_kernel(OperationKind operation, std::size_t n, cudaStream_t stream, const double* a, const double* b, double* c)
{
    constexpr int BLOCK_SIZE {256};
    const auto grid_size {static_cast<unsigned int>((n + std::size_t(BLOCK_SIZE) - 1) / std::size_t(BLOCK_SIZE))};

    switch (operation)
    {
        case OperationKind::Add:
        {
            kernel_add<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Multiply:
        {
            kernel_multiply<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Divide:
        {
            kernel_divide<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Power:
        {
            kernel_power<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Exp:
        {
            kernel_exp<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Log:
        {
            kernel_log<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
        case OperationKind::Sqrt:
        {
            kernel_sqrt<<<grid_size, BLOCK_SIZE, 0, stream>>>(n, a, b, c);
            return cudaGetLastError();
        }
    }

    return cudaErrorInvalidValue;
}



/**
 * @brief Compute the sum of all elements in a vector (serial).
 * @param numbers Vector to sum.
 * @return Sum of elements.
 */
[[nodiscard]]
double check_sum(const std::vector<double>& numbers)
{
    auto sum {0.0};
    for (const auto number : numbers)
    {
        sum += number;
    }
    return sum;
}

} // namespace



/**
 * @brief Entry point into program.
 */
int main(int argc, char** argv)
{
    try
    {
        if (argc < 4)
        {
            THROW_INVALID_ARGUMENT("Usage: cuda.x time_limit vec_size operation");
        }

        const auto test_time_seconds {parse_double(argv[1])};
        const auto n {parse_size(argv[2])};
        const auto operation_string {std::string_view(argv[3])};
        const auto operation {parse_operation(operation_string)};

        if (test_time_seconds <= 0.0)
        {
            THROW_INVALID_ARGUMENT("time_limit must be > 0.");
        }
        if (n == 0)
        {
            THROW_INVALID_ARGUMENT("vec_size must be > 0.");
        }

        std::mt19937_64 rng(RNG_SEED);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        auto numbers_a {std::vector<double>{}};
        auto numbers_b {std::vector<double>{}};
        numbers_a.reserve(n);
        numbers_b.reserve(n);

        for (auto i = std::size_t(0); i < n; i++)
        {
            numbers_a.emplace_back(dist(rng));
            numbers_b.emplace_back(dist(rng));
        }

        auto expected_value {0.0};
        {
            auto numbers_c {std::vector<double>(n)};
            validate_sizes(numbers_a, numbers_b, numbers_c);

            serial_task(operation, numbers_a, numbers_b, numbers_c);
            expected_value = check_sum(numbers_c);

            std::cout << "Serial computed expected value: " << expected_value << "\n";
        }

        // ======= Calculation Starts ========
        const auto t0 {std::chrono::steady_clock::now()};

        int device_id {0};
        cuda_check(cudaGetDevice(&device_id), "cudaGetDevice failed.");

        cudaDeviceProp prop {};
        cuda_check(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties failed.");
        std::cerr << "Using device: " << prop.name << "\n";

        // Results
        auto numbers_c {std::vector<double>(n)};

        // Allocate device memory once
        double* dev_a {};
        double* dev_b {};
        double* dev_c {};

        cuda_check(cudaMalloc(&dev_a, n * sizeof(double)), "cudaMalloc dev_a failed.");
        cuda_check(cudaMalloc(&dev_b, n * sizeof(double)), "cudaMalloc dev_b failed.");
        cuda_check(cudaMalloc(&dev_c, n * sizeof(double)), "cudaMalloc dev_c failed.");

        cudaStream_t stream {};
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate failed.");

        // Copy once
        cuda_check(cudaMemcpyAsync(dev_a, numbers_a.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync dev_a failed.");
        cuda_check(cudaMemcpyAsync(dev_b, numbers_b.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync dev_b failed.");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize after H2D copies failed.");

        // ======= Start up =======
        const auto t1 {std::chrono::steady_clock::now()};
        const auto deadline {t1 + std::chrono::duration<double>(test_time_seconds)};

        auto iters {std::uint64_t(0)};

        // Do as many times as possible before time runs out
        do
        {
            cuda_check(launch_kernel(operation, n, stream, dev_a, dev_b, dev_c), "Kernel launch failed.");
            iters++;
        }
        while (std::chrono::steady_clock::now() < deadline);

        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize after kernels failed.");

        // Copy results back
        cuda_check(cudaMemcpyAsync(numbers_c.data(), dev_c, n * sizeof(double), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H failed.");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize after D2H copy failed.");

        // ======= Clean up =======
        const auto t2 {std::chrono::steady_clock::now()};

        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy failed.");
        cuda_check(cudaFree(dev_a), "cudaFree dev_a failed.");
        cuda_check(cudaFree(dev_b), "cudaFree dev_b failed.");
        cuda_check(cudaFree(dev_c), "cudaFree dev_c failed.");

        // ======= Calculation Ends ========
        const auto t3 {std::chrono::steady_clock::now()};

        const auto calculated_value {check_sum(numbers_c)};

        const auto time_setup {std::chrono::duration<double>(t1 - t0).count()};
        const auto time_calc {std::chrono::duration<double>(t2 - t1).count()};
        const auto time_cleanup {std::chrono::duration<double>(t3 - t2).count()};
        const auto time_total {std::chrono::duration<double>(t3 - t0).count()};
        const auto time_per_iteration {time_calc / static_cast<double>(iters)};

        const auto passed_check {std::abs(calculated_value - expected_value) < 1.0e-9};

        const auto method {std::string("Parallel CUDA")};
        const auto comments {std::string("operation:") + std::string(operation_string)};

        std::cout
            << "method,expected_value,calculated_value,iterations,"
            "time_per_iteration,time_setup,time_calc,time_cleanup,"
            "time_total,passed_check,comments\n";

        std::cout
            << method << ','
            << std::setprecision(17) << expected_value << ','
            << std::setprecision(17) << calculated_value << ','
            << iters << ','
            << std::scientific << std::setprecision(9) << time_per_iteration << ','
            << std::fixed      << std::setprecision(6) << time_setup << ','
            << std::fixed      << std::setprecision(6) << time_calc << ','
            << std::fixed      << std::setprecision(6) << time_cleanup << ','
            << std::fixed      << std::setprecision(6) << time_total << ','
            << passed_check << ','
            << comments
            << '\n';

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
