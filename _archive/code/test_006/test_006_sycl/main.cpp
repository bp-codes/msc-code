#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <sycl/sycl.hpp>

const int nx = 2000;
const int ny = 2000;
const int nt = 1000;
const double alpha = 1.0;
const double dx = 1.0;
const double dy = 1.0;
const double dt = 0.1;

void checkStability() {
    double coeff = dt * alpha * (1.0 / (dx * dx) + 1.0 / (dy * dy));
    if (coeff >= 0.5) {
        std::cerr << "Stability condition violated! Reduce dt." << std::endl;
        std::exit(1);
    }
}

void saveToCSV(const std::vector<std::vector<double>>& u, const std::string& filename) {
    std::ofstream file(filename);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            file << u[i][j];
            if (j != ny - 1) file << ",";
        }
        file << "\n";
    }
}

int main() {
    // List SYCL platforms/devices
    auto platforms = sycl::platform::get_platforms();
    for (const auto& platform : platforms) {
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << "\n";
        for (const auto& device : platform.get_devices()) {
            std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << "\n";
            std::cout << "    Type: "
                      << (device.is_gpu() ? "GPU" : device.is_cpu() ? "CPU" : "Other") << "\n";
        }
    }

    auto start_setup = std::chrono::high_resolution_clock::now();
    checkStability();

    std::vector<std::vector<double>> u(nx, std::vector<double>(ny, 0.0));
    for (int i = nx / 4; i < 3 * nx / 4; ++i)
        for (int j = ny / 4; j < 3 * ny / 4; ++j)
            u[i][j] = 100.0;

    saveToCSV(u, "heat_input.csv");

    auto end_setup = std::chrono::high_resolution_clock::now();
    auto start_calc = std::chrono::high_resolution_clock::now();

    std::vector<double> u_flat(nx * ny, 0.0);
    std::vector<double> u_new_flat(nx * ny, 0.0);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            u_flat[i * ny + j] = u[i][j];

    // Create SYCL queue on GPU (with fallback error)
    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
        std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    } catch (const sycl::exception& e) {
        std::cerr << "GPU queue creation failed: " << e.what() << "\n";
        return 1;
    }

    sycl::buffer<double, 1> u_buf(u_flat.data(), sycl::range<1>(nx * ny));
    sycl::buffer<double, 1> u_new_buf(u_new_flat.data(), sycl::range<1>(nx * ny));

    for (int step = 0; step < nt; ++step) {
        q.submit([&](sycl::handler& h) {
            auto u_acc = u_buf.get_access<sycl::access::mode::read>(h);
            auto u_new_acc = u_new_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<2>(nx - 2, ny - 2), [=](sycl::id<2> idx) {
                int i = idx[0] + 1;
                int j = idx[1] + 1;
                int id = i * ny + j;

                u_new_acc[id] = u_acc[id] + alpha * dt * (
                    (u_acc[(i + 1) * ny + j] - 2.0 * u_acc[id] + u_acc[(i - 1) * ny + j]) / (dx * dx) +
                    (u_acc[i * ny + (j + 1)] - 2.0 * u_acc[id] + u_acc[i * ny + (j - 1)]) / (dy * dy)
                );
            });
        });
        q.wait();
        std::swap(u_buf, u_new_buf);  // swap buffers
    }

    // Copy final result back: 1D → 2D
    auto u_result = u_buf.get_host_access();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            u[i][j] = u_result[i * ny + j];

    auto end_calc = std::chrono::high_resolution_clock::now();
    saveToCSV(u, "heat_output.csv");

    std::cout << "Simulation complete. Output saved to heat_output.csv\n";

    std::chrono::duration<double> t_setup = end_setup - start_setup;
    std::chrono::duration<double> t_compute = end_calc - start_calc;
    std::cout << "Setup Time (s): " << t_setup.count() << "\n";
    std::cout << "Compute Time (s): " << t_compute.count() << "\n";

    return 0;
}
