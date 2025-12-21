#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

class Vec3 {
private:
    double x, y, z;

public:
    // Constructor
    Vec3(double x = 0.0, double y = 0.0, double z = 0.0)
        : x(x), y(y), z(z) {}

    // Getters
    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }

    // Setters
    void setX(double xVal) { x = xVal; }
    void setY(double yVal) { y = yVal; }
    void setZ(double zVal) { z = zVal; }

    // Utility to print vector
    void print() const {
        std::cout << "Vec3(" << x << ", " << y << ", " << z << ")" << std::endl;
    }

    // Friend function to calculate distance between two Vec3 objects
    friend double distance(const Vec3& a, const Vec3& b);
};

// Definition of the friend function
double distance(const Vec3& a, const Vec3& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Measurement struct to store values at each point
struct Measurement {
    double density {};
    double energy {};
};

void calculate() {
    std::vector<Vec3> points;

    const int count = 10000;

    points.reserve(count);

    const double minVal = 0.0;
    const double maxVal = 10.0;

    for (int i = 0; i < count; ++i) {
        double t = static_cast<double>(i) / (count - 1);
        double x = minVal + t * (maxVal - minVal);
        double y = minVal + t * (maxVal - minVal);
        double z = minVal + t * (maxVal - minVal);
        points.emplace_back(x, y, z);
    }

    // Initialize measurements for each point
    std::vector<Measurement> measurements(points.size());


    // Calculate neighbor list for each point
    const double cutoff = 20.0; // example cutoff distance

    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            double separation = distance(points[i], points[j]);
            if (separation < cutoff) {

                // Calculate density
                double density = 0.01 * std::exp(-0.1 * (separation / 20.0));
                measurements[i].density = measurements[i].density + density;
                measurements[j].density = measurements[j].density + density;

                // Calculate energy
                double energy = 0.1 * std::pow(density, 0.1) + 0.3 * std::pow(density, 0.74) - 0.2 * std::pow(density, 0.12);
                measurements[i].energy = measurements[i].energy + energy;
                measurements[j].energy = measurements[j].energy + energy;

            }
        }
    }

    // Print number of neighbors for first few points
    for (int i = 0; i < 5; ++i) {
        std::cout << "Point " << i << " density: " << measurements[i].density << " energy: " << measurements[i].energy << std::endl;
    }
}

int main() {

    // run calculation
    auto start_calc = std::chrono::high_resolution_clock::now();
    calculate();
    auto end_calc = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_calc.count() << std::endl;

    return 0;
}
