// gpu_reuse.cpp
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <sycl/sycl.hpp>

// helper: round n up to multiple of m
static inline std::size_t round_up(std::size_t n, std::size_t m) 
{
    return ((n + m - 1) / m) * m;
}


// ---------------------------------------------------------------------------
// largest_power_of_two_leq
// 
// Returns the largest power of two that is less than or equal to n.
// 
// Example:
//   largest_power_of_two_leq(100) → 64
//   largest_power_of_two_leq(64)  → 64
//   largest_power_of_two_leq(65)  → 64
//   largest_power_of_two_leq(0)   → 0
// 
// Algorithm: Bit-smearing (fill all bits to the right of the highest set bit)
// ---------------------------------------------------------------------------
inline std::size_t largest_power_of_two_leq(std::size_t n) noexcept
{
    if (n == 0) return 0;

    // Fill all bits to the right of the highest 1-bit
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if constexpr (sizeof(std::size_t) == 8) {
        n |= n >> 32;
    }

    // n is now 2^k - 1 (e.g., 127 for 64), so n - (n >> 1) = 2^{k-1}
    return n - (n >> 1);
}


double sycl_task(const double* d_numbers, std::size_t numbers_size, sycl::queue& q,
                    const std::size_t work_group_size_limit)
{
    if (numbers_size == 0) return 0.0;

    auto result {0.0};

    // local_size - multiple of 32 for Nvidia, 64 for AMD - work items per work group
    const auto max_work_group  = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    const auto local_size  = std::min<std::size_t>(work_group_size_limit, max_work_group);
    
    // Total number of work-items
    const auto work_groups = std::size_t((numbers_size + local_size - 1) / local_size);
    const auto global_size = std::size_t(work_groups * local_size);

    {
        // Alias result and out_buf (a 1 element buffer)
        sycl::buffer<double> out_buf(&result, sycl::range<1>(1));


        q.submit([&](sycl::handler& h) 
        {

            // Set up reduction to pass to kernel
            auto reduction_argument = sycl::reduction(
                out_buf, h,
                sycl::plus<double>{},
                sycl::property::reduction::initialize_to_identity{}
            );

            // Launch a kernel for "work_groups" number of work groups
            h.parallel_for( 
                            sycl::nd_range<1>(global_size, local_size), 
                            reduction_argument,
                            [=](sycl::nd_item<1> it, auto& accumulator) 
            {
                // Total number of work-items in the 1D grid.
                const auto grid_work_items = std::size_t(it.get_global_range(0));

                // Linear global ID of this work-item.
                const auto grid_id = std::size_t(it.get_global_linear_id());

                // Thread-local partial sum.
                auto thread_partial_sum {0.0};

                // Add to thread_partial_sum in strides of grid_work_items
                for (std::size_t i = grid_id; i < numbers_size; i += grid_work_items)
                {
                    thread_partial_sum += d_numbers[i];
                }

                // Contribute this work-item's partial sum to the global reduction.
                accumulator.combine(thread_partial_sum);
            });

        }).wait();
    }

    return result;
}



// Serial task - sum numbers in the vector
double serial_task(const std::vector<double>& numbers)
{
    auto sum {0.0};
    for(const auto val : numbers)
    {
        sum += val;
    }
    return sum;
}



int main(int argc, char** argv) 
{

    // Must have 4 arguments
    if (argc < 4) 
    {
        std::cerr << "Usage: " << argv[0] << "   time_limit   vec_size   device\n";
        return 1;
    }

    // Read in test_time and size of vector
    const double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);
    const std::size_t work_group_size_limit = (argc < 4) ? 256 : std::atoi(argv[3]);
    std::string device_selection = argv[4];
    std::transform(device_selection.begin(), device_selection.end(), device_selection.begin(), ::tolower);
    const std::string operation = "Sum vector elements.";


    if(N <= 0)
    {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }

    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);  // [0.0, 1.0)

    // Vector of numbers
    std::vector<double> numbers;
    numbers.reserve(N);

    // Populate vector
    for (int i = 0; i < N; ++i) 
    {
        numbers.emplace_back(dist(rng));
    }

    // Calculate expected value using the serial version
    auto expected_value = serial_task(numbers);
 

    // ======= Calculation Starts ========
    
    auto t0 = std::chrono::steady_clock::now();

    // SYCL Device selector code
    auto make_queue = [&](const std::string& device_selection) -> sycl::queue
    {
        try
        {
            if (device_selection == "cpu") 
            {    
                // Can't find AdaptiveCPP OpenMP
                for (const auto& p : sycl::platform::get_platforms())
                {
                    for (const auto& d : p.get_devices())
                    {
                        if (d.is_cpu()) 
                        {
                            return sycl::queue{d};
                        }
                    }
                }

                // Unable to find CPU - throw
                throw std::runtime_error("No CPU device found.");
            }
            else if (device_selection == "gpu") 
            {
                return sycl::queue{sycl::gpu_selector_v};
            }
            return sycl::queue{sycl::default_selector_v};
        }
        catch (const sycl::exception& e)
        {   
            std::cout << "Unable to select, falling back to default device." << std::endl;
            std::cout << "  Response: " << e.what() << std::endl;
            return sycl::queue{sycl::default_selector_v};
        }        
    };

    // Set up queue, select device
    auto q = make_queue(device_selection);

    std::cerr << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Allocate device memory once
    double* d_numbers = sycl::malloc_device<double>(N, q);

    // Copy once
    q.memcpy(d_numbers, numbers.data(), N * sizeof(double)).wait();

    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = 0;

    // Reuse d_numbers on every iteration
    auto calculated_value {0.0};

    do 
    {
        calculated_value = sycl_task(d_numbers, N, q, work_group_size_limit);
        iters++;
    } 
    while (std::chrono::steady_clock::now() < deadline);

    // Clean up
    auto t2 = std::chrono::steady_clock::now();    

    sycl::free(d_numbers, q); // free at end

    // Actual end time
    auto t3 = std::chrono::steady_clock::now();

    // ======= Calculation Ends ========
   
    auto time_setup = std::chrono::duration<double>(t1 - t0).count();
    auto time_calc = std::chrono::duration<double>(t2 - t1).count();
    auto time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    auto time_total = std::chrono::duration<double>(t3 - t0).count();
    auto time_per_iteration = time_calc / iters;

    std::string method {"SYCL"};
    std::string device {device_selection};
    std::string comments {"operation:" + operation};
    bool passed_check = std::abs(calculated_value - expected_value) < 1.0e-6;


    std::cout << method << "," 
              << device << ","               
              << std::scientific << std::setprecision(12)
              << expected_value << "," 
              << calculated_value << "," 
              << std::scientific << std::setprecision(6)
              << iters << "," 
              << time_per_iteration << "," 
              << time_setup << "," 
              << time_calc << "," 
              << time_cleanup << "," 
              << time_total << "," 
              << passed_check << "," 
              << comments << "," 
              << work_group_size_limit << ""
              << std::endl;

    return 0;
}
