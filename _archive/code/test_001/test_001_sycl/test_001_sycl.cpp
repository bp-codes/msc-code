#include <iostream>
#include <cmath>
#include <chrono>
#include <CL/sycl.hpp>

int main() {

    auto start_setup = std::chrono::high_resolution_clock::now();
    //############################################################

    constexpr int num_elements = 100'000'000;
    
    auto end_setup = std::chrono::high_resolution_clock::now();




    auto start_calc = std::chrono::high_resolution_clock::now();
    //############################################################

    // Allocate host memory
    std::vector<double> result(num_elements);


    auto platforms = sycl::platform::get_platforms();
    for (const auto& platform : platforms) {
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << "\n";
        for (const auto& device : platform.get_devices()) {
            std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << "\n";
            std::cout << "    Type: "
                      << (device.is_gpu() ? "GPU" : device.is_cpu() ? "CPU" : "Other") << "\n";
        }
    }

    // Create SYCL queue on GPU (with fallback error)
    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
        std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    } catch (const sycl::exception& e) {
        std::cerr << "GPU queue creation failed: " << e.what() << "\n";
        return 1;
    }


    //std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Create buffers
    {
        sycl::buffer<double> result_buf(result.data(), sycl::range<1>(num_elements));

        q.submit([&](sycl::handler& h) {
            auto res = result_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(num_elements), [=](sycl::id<1> i) {
                double x = static_cast<double>(i[0] % 100) / 10.0;
                res[i] = sycl::exp(x); // device math function
            });
        });
    }


    // Sum results on host
    double sum = 0.0;
    for (double val : result)
    {
        sum += val;
    }


    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();
    

    // Calculate duration
    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << sum << std::endl;

    return 0;
}

