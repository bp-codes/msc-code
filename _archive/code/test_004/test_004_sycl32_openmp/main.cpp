#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <sycl/sycl.hpp>
#include <omp.h> // OpenMP

// Host-side Vec3 class
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

// Device-compatible vector
struct SyclVec3 {
    float x, y, z;
};

template <>
struct sycl::is_device_copyable<SyclVec3> : std::true_type {};

SyclVec3 to_sycl_vec3(const Vec3& v) {
    return SyclVec3{
        static_cast<float>(v.getX()),
        static_cast<float>(v.getY()),
        static_cast<float>(v.getZ())
    };
}

int main() {
    auto start_setup = std::chrono::high_resolution_clock::now();

    constexpr int num_iterations = 301;
    const int total_points = num_iterations * num_iterations * num_iterations;
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

    std::vector<SyclVec3> reference_points_device;
    for (const auto& v : reference_points)
        reference_points_device.push_back(to_sycl_vec3(v));

    std::vector<SyclVec3> points_device;
    for (const auto& v : points)
        points_device.push_back(to_sycl_vec3(v));

    int split = total_points * 9 / 10;
    sycl::queue q;

    // GPU kernel for first 75%
    {
        sycl::buffer<SyclVec3, 1> points_buf(points_device.data(), sycl::range<1>(split));
        sycl::buffer<SyclVec3, 1> refs_buf(reference_points_device.data(), sycl::range<1>(reference_points_device.size()));
        sycl::buffer<double, 1> result_buf(calculated_results.data(), sycl::range<1>(split));

        q.submit([&](sycl::handler& h) {
            auto points_acc = points_buf.get_access<sycl::access::mode::read>(h);
            auto refs_acc = refs_buf.get_access<sycl::access::mode::read>(h);
            auto res_acc = result_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(split), [=](sycl::id<1> idx) {
                float acc = 0.0f;
                SyclVec3 p = points_acc[idx];
                for (size_t r = 0; r < refs_acc.get_range()[0]; ++r) {
                    SyclVec3 ref = refs_acc[r];
                    float dx = p.x - ref.x;
                    float dy = p.y - ref.y;
                    float dz = p.z - ref.z;
                    float dist = sycl::sqrt(dx * dx + dy * dy + dz * dz);

                    float term_1 = 0.5f * sycl::exp(-0.1f * dist);
                    float term_2 = sycl::pow(dist, 0.33f);
                    float term_3 = 0.25f * dist * dist * dist - 2.5f * dist * dist - 0.3f * dist + 3.2f;
                    float term_4 = 0.22f * sycl::pow(dist, -1.5f);
                    float term_5 = 0.1f * sycl::pow(dist, -2.5f);
                    float term_6 = -0.2f * sycl::pow(dist, -3.5f);
                    float term_7 = -0.8f * sycl::pow(dist, 0.5f);

                    acc += 1.0e-4f * (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7);
                }
                res_acc[idx] = static_cast<double>(acc);
            });
        });
    }

    // CPU OpenMP parallel loop for remaining 25%
    #pragma omp parallel for
    for (int idx = split; idx < total_points; ++idx) {
        const Vec3& p = points[idx];
        double acc = 0.0;
        for (const Vec3& ref : reference_points) {
            double dx = p.getX() - ref.getX();
            double dy = p.getY() - ref.getY();
            double dz = p.getZ() - ref.getZ();
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            double term_1 = 0.5 * std::exp(-0.1 * dist);
            double term_2 = std::pow(dist, 0.33);
            double term_3 = 0.25 * dist * dist * dist - 2.5 * dist * dist - 0.3 * dist + 3.2;
            double term_4 = 0.22 * std::pow(dist, -1.5);
            double term_5 = 0.1 * std::pow(dist, -2.5);
            double term_6 = -0.2 * std::pow(dist, -3.5);
            double term_7 = -0.8 * std::pow(dist, 0.5);

            acc += 1.0e-4 * (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7);
        }
        calculated_results[idx] = acc;
    }

    q.wait();  // Wait for GPU
    auto end_calc = std::chrono::high_resolution_clock::now();

    for (const double r : calculated_results)
        result += r;

    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << result << std::endl;

    return 0;
}
