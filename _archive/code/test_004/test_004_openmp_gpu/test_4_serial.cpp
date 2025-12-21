#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>



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


int main() 
{


    auto start_setup = std::chrono::high_resolution_clock::now();

    constexpr int num_iterations = 301;
    double result = 0.0;

    std::vector<Vec3> reference_points {};

    Vec3 reference1 {-3.1550, 1.2443, 12.455};
    Vec3 reference2 {1.75, 6.23, 15.97};
    Vec3 reference3 {-7.822, 2.5541, -25.21};
    Vec3 reference4 {-0.355, 1.412, -19.8};
    Vec3 reference5 {3.51,-7.55,25.9};

    reference_points.emplace_back(reference1);
    reference_points.emplace_back(reference2);
    reference_points.emplace_back(reference3);
    reference_points.emplace_back(reference4);
    reference_points.emplace_back(reference5);



    std::vector<double> extent {-10.0, 10.0, -10.0, 10.0, -10.0, 10.0};
    std::vector<Vec3> points {};
    std::vector<double> calculated_results {};


    for (int i = 0; i < num_iterations; ++i) 
    {
        double x = extent[0] + i * (extent[1] - extent[0]) / (num_iterations - 1);

        for (int j = 0; j < num_iterations; ++j) 
        {
            double y = extent[2] + j * (extent[3] - extent[2]) / (num_iterations - 1);

            for (int k = 0; k < num_iterations; ++k) 
            {
                double z = extent[4] + k * (extent[5] - extent[4]) / (num_iterations - 1);
                Vec3 point {x, y, z};
                points.emplace_back(point);
                calculated_results.emplace_back(0.0);
            }
        }
    }

    auto end_setup = std::chrono::high_resolution_clock::now();


    auto start_calc = std::chrono::high_resolution_clock::now();
    //############################################################
    // Code to Parallelize starts here

    
    int idx {0};
    for (const auto& point : points) 
    {
        calculated_results[idx] = 0.0;

        for (const auto& reference_point : reference_points) 
        {

            double r = distance(point, reference_point);

            double term_1 = 0.5 * exp(-0.1 * r);
            double term_2 = std::pow(r, 0.33);
            double term_3 = 0.25 * r*r*r - 2.5 * r * r - 0.3 * r + 3.2;
            double term_4 = 0.22 * std::pow(r, -1.5);
            double term_5 = 0.1 * std::pow(r, -2.5);
            double term_6 = -0.2 * std::pow(r, -3.5);
            double term_7 = -0.8 * std::pow(r, 0.5);

            calculated_results[idx] = calculated_results[idx] + 1.0e-4 * (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7);

        }

        idx = idx + 1;
    }


    // Code to Parallelize ends here
    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();

    for(const double r : calculated_results)
    {
        result = result + r;
    }
    
    // Calculate duration
    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << result << std::endl;

    return 0;
}

