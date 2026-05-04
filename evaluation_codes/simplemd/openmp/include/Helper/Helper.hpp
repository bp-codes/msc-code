#ifndef HELPER_HPP
#define HELPER_HPP


#include <charconv>
#include <chrono>
#include <concepts>
#include <sys/resource.h>
#include <cstdint>
#include <random>
#include <string>
#include <ranges>
#include "Helper/Error.hpp"
#include "json.hpp"



namespace helper
{


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



std::size_t get_num_threads()
{
    const char* env = std::getenv("NUM_THREADS");

    if (env != nullptr)
    {
        return static_cast<std::size_t>(std::stoul(env));
    }

    return 6;
}



std::uint64_t max_rss_kb()
{
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);

#if defined(__APPLE__)
    // macOS reports bytes
    return usage.ru_maxrss / 1024;
#else
    // Linux reports kilobytes
    return usage.ru_maxrss;
#endif
}



[[nodiscard]]
std::string random_suffix(const std::size_t n)
{
    static constexpr char charset[] =
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789";

    static thread_local std::mt19937 rng{std::random_device{}()};
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
double parse_floating_point(const char* s)
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



/**
 * @brief Returns a calue converted to string.
 * @param value to convert to string.
 * @return string.
 */
template<typename T>
requires (std::floating_point<T>)
[[nodiscard]]
std::string to_string_precise(const T& value)
{
    std::ostringstream oss;
    oss << std::scientific
        << std::setprecision(std::numeric_limits<T>::max_digits10)
        << value;
    return oss.str();
}



template<typename Range>
requires std::ranges::input_range<Range> &&
         std::floating_point<std::ranges::range_value_t<Range>>
[[nodiscard]]
nlohmann::json
to_string_precise_vector(const Range& values_in)
{
    auto values = nlohmann::json::array();

    for (const auto& v : values_in)
    {
        values.emplace_back(helper::to_string_precise(v));
    }

    return values;
}



/**
 * @brief Compute the sum of all elements in a vector (serial).
 * @param numbers Vector to sum.
 * @return Sum of elements.
 */
template<typename T>
[[nodiscard]]
T check_sum(const std::vector<T>& numbers)
{
    return std::accumulate(numbers.begin(), numbers.end(), 0.0);
}








}


#endif
