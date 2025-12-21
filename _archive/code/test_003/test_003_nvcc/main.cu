#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

struct Vec3 {
    double x, y, z;
};

struct Measurement {
    double density;
    double energy;
};

__device__ double distance(const Vec3& a, const Vec3& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__global__ void computeMeasurements(Vec3* d_points, Measurement* d_measurements, int count, double cutoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    for (int j = i + 1; j < count; ++j) {
        double separation = distance(d_points[i], d_points[j]);
        if (separation < cutoff) {
            double density = 0.01 * exp(-0.1 * (separation / 20.0));
            double energy = 0.1 * pow(density, 0.1) + 0.3 * pow(density, 0.74) - 0.2 * pow(density, 0.12);

            atomicAdd(&d_measurements[i].density, density);
            atomicAdd(&d_measurements[j].density, density);
            atomicAdd(&d_measurements[i].energy, energy);
            atomicAdd(&d_measurements[j].energy, energy);
        }
    }
}

void calculate() {
    const int count = 10000;
    std::vector<Vec3> h_points(count);
    std::vector<Measurement> h_measurements(count);

    const double minVal = 0.0;
    const double maxVal = 10.0;

    for (int i = 0; i < count; ++i) {
        double t = static_cast<double>(i) / (count - 1);
        double val = minVal + t * (maxVal - minVal);
        h_points[i] = {val, val, val};
        h_measurements[i] = {0.0, 0.0};
    }

    Vec3* d_points;
    Measurement* d_measurements;

    cudaMalloc(&d_points, count * sizeof(Vec3));
    cudaMalloc(&d_measurements, count * sizeof(Measurement));
    cudaMemcpy(d_points, h_points.data(), count * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_measurements, h_measurements.data(), count * sizeof(Measurement), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    computeMeasurements<<<blocks, threadsPerBlock>>>(d_points, d_measurements, count, 20.0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_measurements.data(), d_measurements, count * sizeof(Measurement), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_measurements);

    for (int i = 0; i < 5; ++i) {
        std::cout << "Point " << i << " density: " << h_measurements[i].density
                  << " energy: " << h_measurements[i].energy << std::endl;
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    calculate();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds\n";
    return 0;
}