#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Vec3 {
    double x, y, z;
    __host__ __device__ Vec3(double x = 0.0, double y = 0.0, double z = 0.0)
        : x(x), y(y), z(z) {}
};

__host__ __device__ double distance(const Vec3& a, const Vec3& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__global__ void compute_kernel(const Vec3* points, double* results, int num_points, Vec3* ref_points, int num_refs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    double total = 0.0;
    Vec3 point = points[idx];

    for (int j = 0; j < num_refs; ++j) {
        double r = distance(point, ref_points[j]);

        double term_1 = 0.5 * exp(-0.1 * r);
        double term_2 = pow(r, 0.33);
        double term_3 = 0.25 * r * r * r - 2.5 * r * r - 0.3 * r + 3.2;
        double term_4 = 0.22 * pow(r, -1.5);
        double term_5 = 0.1 * pow(r, -2.5);
        double term_6 = -0.2 * pow(r, -3.5);
        double term_7 = -0.8 * pow(r, 0.5);

        //printf("%f,%f,%f,%f,%f,%f,%f,%f\n",r,term_1,term_2,term_3,term_4,term_5,term_6,term_7);
        total += 1.0e-4 * (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7);
    }
    //printf("Debug %f \n",total);
    results[idx] = total;
}

int main() {
    constexpr int num_iterations = 301;
    const int num_points = num_iterations * num_iterations * num_iterations;

    std::vector<Vec3> host_points;
    host_points.reserve(num_points);
    std::vector<double> host_results(num_points, 0.0);

    std::vector<Vec3> ref_points = {
        Vec3{-3.1550, 1.2443, 12.455},
        Vec3{1.75, 6.23, 15.97},
        Vec3{-7.822, 2.5541, -25.21},
        Vec3{-0.355, 1.412, -19.8},
        Vec3{3.51, -7.55, 25.9}
    };


    auto start_setup = std::chrono::high_resolution_clock::now();
    std::vector<double> extent{-10.0, 10.0, -10.0, 10.0, -10.0, 10.0};

    for (int i = 0; i < num_iterations; ++i) {
        double x = extent[0] + i * (extent[1] - extent[0]) / (num_iterations - 1);
        for (int j = 0; j < num_iterations; ++j) {
            double y = extent[2] + j * (extent[3] - extent[2]) / (num_iterations - 1);
            for (int k = 0; k < num_iterations; ++k) {
                double z = extent[4] + k * (extent[5] - extent[4]) / (num_iterations - 1);
                host_points.emplace_back(x, y, z);
            }
        }
    }
    auto end_setup = std::chrono::high_resolution_clock::now();

    thrust::device_vector<Vec3> dev_points = host_points;
    thrust::device_vector<Vec3> dev_refs = ref_points;
    thrust::device_vector<double> dev_results(num_points, 0.0);

    auto start_calc = std::chrono::high_resolution_clock::now();
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dev_points.data()),
        thrust::raw_pointer_cast(dev_results.data()),
        num_points,
        thrust::raw_pointer_cast(dev_refs.data()),
        static_cast<int>(ref_points.size())
    );
    cudaDeviceSynchronize();
    auto end_calc = std::chrono::high_resolution_clock::now();

    thrust::copy(dev_results.begin(), dev_results.end(), host_results.begin());

    double result = 0.0;
    for (double r : host_results) {
        result += r;
    }

    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << result << std::endl;

    return 0;
}
