// gpu_reuse_buffer_inorder.cpp
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <cstddef>


// Built-in reduction using a 1-element buffer, with an in-order queue.
// No events / depends_on needed.
inline void sycl_task_submit(const double* data,
                            std::size_t N,
                            sycl::queue& q,
                            double* sum)
{
  q.parallel_for(
      sycl::range<1>(N),
      sycl::reduction(sum, sycl::plus<>(),
                      sycl::property::reduction::initialize_to_identity{}),
      [=](sycl::id<1> i, auto& r) {
        r += data[i];          // read-only on data
      });
}



// Serial task - sum numbers in the vector
double serial_task(const std::vector<double>& numbers)
{
    double sum = 0.0;
    for (double v : numbers) sum += v;
    return sum;
}

int main(int argc, char** argv)
{
    // prog + 4 params
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " time_limit vec_size work_group_limit device\n";
        return 1;
    }

    const double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);
    const std::size_t work_group_size_limit = static_cast<std::size_t>(std::atoi(argv[3]));
    std::string device_selection = argv[4];
    std::transform(device_selection.begin(), device_selection.end(),
                   device_selection.begin(), ::tolower);

    const std::string operation = "Sum vector elements.";

    if (N <= 0)
    {
        std::cerr << "vec_size must be > 0\n";
        return 1;
    }

    // Random number generator
    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> numbers;
    numbers.reserve(N);
    for (int i = 0; i < N; ++i)
    {
        numbers.emplace_back(dist(rng));
    }

    const double expected_value = serial_task(numbers);

    // ======= Calculation Starts ========
    auto t0 = std::chrono::steady_clock::now();

    auto make_queue = [&](const std::string& sel) -> sycl::queue
    {
        sycl::property_list props{sycl::property::queue::in_order{}};

        try
        {
            if (sel == "cpu")
            {
                for (const auto& p : sycl::platform::get_platforms())
                {
                    for (const auto& d : p.get_devices())
                    {
                        if (d.is_cpu())
                            return sycl::queue{d, props};
                    }
                }
                throw std::runtime_error("No CPU device found.");
            }
            else if (sel == "gpu")
            {
                return sycl::queue{sycl::gpu_selector_v, props};
            }
            return sycl::queue{sycl::default_selector_v, props};
        }
        catch (const sycl::exception& e)
        {
            std::cout << "Unable to select, falling back to default device.\n";
            std::cout << "  Response: " << e.what() << "\n";
            return sycl::queue{sycl::default_selector_v, props};
        }
    };

    sycl::queue q = make_queue(device_selection);

    std::cerr << "Using device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Keep output buffer once
    double* data = sycl::malloc_shared<double>(N, q);
    double* sum  = sycl::malloc_shared<double>(1, q);

    auto t1 = std::chrono::steady_clock::now();
    auto deadline = t1 + std::chrono::duration<double>(test_time);

    std::uint64_t iters = 0;
    double calculated_value = 0.0;


    for (size_t i = 0; i < N; ++i) 
    {
        data[i] = numbers[i];
    }
    *sum = 0.0;

    do
    {
        sycl_task_submit(data, static_cast<std::size_t>(N), q, sum);
        calculated_value = *sum;
        ++iters;
    }
    while (std::chrono::steady_clock::now() < deadline);

    // In-order queue still needs a wait before host read
    q.wait();


    auto t2 = std::chrono::steady_clock::now();

    sycl::free(sum, q);
    sycl::free(data, q);

    auto t3 = std::chrono::steady_clock::now();
    // ======= Calculation Ends ========

    auto time_setup   = std::chrono::duration<double>(t1 - t0).count();
    auto time_calc    = std::chrono::duration<double>(t2 - t1).count();
    auto time_cleanup = std::chrono::duration<double>(t3 - t2).count();
    auto time_total   = std::chrono::duration<double>(t3 - t0).count();
    auto time_per_iteration = time_calc / static_cast<double>(iters);

    bool passed_check = std::abs(calculated_value - expected_value) < 1.0e-6;

    std::cout << "SYCL USM" << ","
              << device_selection << ","
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
              << ("operation:" + operation) << ","
              << work_group_size_limit
              << "\n";

    return 0;
}
