#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <sycl/sycl.hpp>

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

    friend double distance(const Vec3& a, const Vec3& b);
};

double distance(const Vec3& a, const Vec3& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Device-compatible 16-bit vector
struct SyclVec3 {
    sycl::half x, y, z;
};

template <>
struct sycl::is_device_copyable<SyclVec3> : std::true_type {};

SyclVec3 to_sycl_vec3(const Vec3& v) {
    return SyclVec3{
        sycl::half(static_cast<float>(v.getX())),
        sycl::half(static_cast<float>(v.getY())),
        sycl::half(static_cast<float>(v.getZ()))
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

    sycl::queue q;
    {
        sycl::buffer<SyclVec3, 1> points_buf(points_device.data(), sycl::range<1>(total_points));
        sycl::buffer<SyclVec3, 1> refs_buf(reference_points_device.data(), sycl::range<1>(reference_points_device.size()));
        sycl::buffer<double, 1> result_buf(calculated_results.data(), sycl::range<1>(total_points));

        q.submit([&](sycl::handler& h) {
            auto points_acc = points_buf.get_access<sycl::access::mode::read>(h);
            auto refs_acc = refs_buf.get_access<sycl::access::mode::read>(h);
            auto res_acc = result_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(total_points), [=](sycl::id<1> idx) {
                sycl::half acc = sycl::half(0.0f);
                SyclVec3 p = points_acc[idx];

                for (size_t r = 0; r < refs_acc.get_range()[0]; ++r) {
                    SyclVec3 ref = refs_acc[r];
                    sycl::half dx = p.x - ref.x;
                    sycl::half dy = p.y - ref.y;
                    sycl::half dz = p.z - ref.z;
                    sycl::half dist = sycl::sqrt(dx * dx + dy * dy + dz * dz);

                    sycl::half term_1 = sycl::half(0.5f) * sycl::exp(sycl::half(-0.1f) * dist);
                    sycl::half term_2 = sycl::pow(dist, sycl::half(0.33f));
                    sycl::half term_3 = sycl::half(0.25f) * dist * dist * dist
                                      - sycl::half(2.5f) * dist * dist
                                      - sycl::half(0.3f) * dist
                                      + sycl::half(3.2f);
                    sycl::half term_4 = sycl::half(0.22f) * sycl::pow(dist, sycl::half(-1.5f));
                    sycl::half term_5 = sycl::half(0.1f) * sycl::pow(dist, sycl::half(-2.5f));
                    sycl::half term_6 = sycl::half(-0.2f) * sycl::pow(dist, sycl::half(-3.5f));
                    sycl::half term_7 = sycl::half(-0.8f) * sycl::pow(dist, sycl::half(0.5f));

                    acc += sycl::half(1.0e-4f) *
                        (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7);
                }

                res_acc[idx] = static_cast<double>(acc);
            });
        });
        q.wait();
    }

    auto end_calc = std::chrono::high_resolution_clock::now();

    for (const double r : calculated_results)
        result += r;

    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << result << std::endl;

    return 0;
}
