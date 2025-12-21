#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>

class Vec3 {
private:
    double x, y, z;

public:
    Vec3(double x = 0.0, double y = 0.0, double z = 0.0)
        : x(x), y(y), z(z) {}

    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }

    void setX(double xVal) { x = xVal; }
    void setY(double yVal) { y = yVal; }
    void setZ(double zVal) { z = zVal; }

    friend double distance(const Vec3& a, const Vec3& b);
};

double distance(const Vec3& a, const Vec3& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void compute_range(const std::vector<Vec3>& points,
                   const std::vector<Vec3>& reference_points,
                   std::vector<double>& calculated_results,
                   size_t start_idx, size_t end_idx) {
    for (size_t idx = start_idx; idx < end_idx; ++idx) {
        double acc = 0.0;
        const Vec3& point = points[idx];

        for (const Vec3& reference_point : reference_points) {
            double r = distance(point, reference_point);

            double term_1 = 0.5 * std::exp(-0.1 * r);
            double term_2 = std::pow(r, 0.33);
            double term_3 = 0.25 * r * r * r - 2.5 * r * r - 0.3 * r + 3.2;
            double term_4 = 0.22 * std::pow(r, -1.5);
            double term_5 = 0.1 * std::pow(r, -2.5);
            double term_6 = -0.2 * std::pow(r, -3.5);
            double term_7 = -0.8 * std::pow(r, 0.5);

            acc += 1.0e-4 * (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7);
        }

        calculated_results[idx] = acc;
    }
}

int main() {
    auto start_setup = std::chrono::high_resolution_clock::now();

    constexpr int num_iterations = 301;
    double result = 0.0;

    std::vector<Vec3> reference_points = {
        {-3.1550, 1.2443, 12.455},
        {1.75, 6.23, 15.97},
        {-7.822, 2.5541, -25.21},
        {-0.355, 1.412, -19.8},
        {3.51, -7.55, 25.9}
    };

    std::vector<double> extent = {-10.0, 10.0, -10.0, 10.0, -10.0, 10.0};
    std::vector<Vec3> points;
    std::vector<double> calculated_results;

    for (int i = 0; i < num_iterations; ++i) {
        double x = extent[0] + i * (extent[1] - extent[0]) / (num_iterations - 1);
        for (int j = 0; j < num_iterations; ++j) {
            double y = extent[2] + j * (extent[3] - extent[2]) / (num_iterations - 1);
            for (int k = 0; k < num_iterations; ++k) {
                double z = extent[4] + k * (extent[5] - extent[4]) / (num_iterations - 1);
                points.emplace_back(x, y, z);
                calculated_results.emplace_back(0.0);
            }
        }
    }

    auto end_setup = std::chrono::high_resolution_clock::now();
    auto start_calc = std::chrono::high_resolution_clock::now();

    // Multithreaded execution
    const size_t total_points = points.size();
    const unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    size_t chunk_size = (total_points + num_threads - 1) / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, total_points);

        threads.emplace_back(compute_range,
                             std::ref(points),
                             std::ref(reference_points),
                             std::ref(calculated_results),
                             start_idx, end_idx);
    }

    for (auto& thread : threads)
        thread.join();

    auto end_calc = std::chrono::high_resolution_clock::now();

    for (const double r : calculated_results)
        result += r;

    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << result << std::endl;

    return 0;
}
