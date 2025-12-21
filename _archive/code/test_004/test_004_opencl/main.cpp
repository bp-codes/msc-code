#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <CL/cl.h>
#include <fstream>
#include <cassert>


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


struct CLVec3 {
    double x, y, z;
};


const char* kernel_file = "kernel.cl";

std::string load_kernel(const char* filename) {
    std::ifstream file(filename);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
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

    // Load kernel source
    std::string kernel_source = load_kernel(kernel_file);
    const char* kernel_src = kernel_source.c_str();

    //
    cl_platform_id selected_platform = nullptr;
    cl_device_id selected_device = nullptr;
    cl_int err;

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    //std::cout << "Platforms: " << num_platforms << std::endl;

    // Try to find a GPU on any platform
    for (cl_uint i = 0; i < num_platforms; ++i) {
        cl_uint num_devices = 0;
        cl_int res = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &selected_device, &num_devices);
        if (res == CL_SUCCESS && num_devices > 0) {
            selected_platform = platforms[i];
            break;
        }
    }

    // Fallback to CPU if no GPU found
    if (!selected_device) {
        std::cerr << "No GPU found, falling back to CPU." << std::endl;
        for (cl_uint i = 0; i < num_platforms; ++i) {
            cl_uint num_devices = 0;
            cl_int res = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &selected_device, &num_devices);
            if (res == CL_SUCCESS && num_devices > 0) {
                selected_platform = platforms[i];
                break;
            }
        }
    }

    if (!selected_device || !selected_platform) {
        std::cerr << "No OpenCL device found." << std::endl;
        return 1;
    }



    // Step 2: Create context and queue
    cl_context context = clCreateContext(nullptr, 1, &selected_device, nullptr, nullptr, &err);
    assert(err == CL_SUCCESS);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, selected_device, 0, &err);
    assert(err == CL_SUCCESS);

    // Step 3: Build program
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_src, nullptr, &err);
    assert(err == CL_SUCCESS);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);


    // Print build log on error
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, selected_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, selected_device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        assert(false);
    }

    // Step 4: Create kernel
    cl_kernel kernel = clCreateKernel(program, "compute", &err);
    assert(err == CL_SUCCESS);



    std::vector<CLVec3> cl_points(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        cl_points[i] = { points[i].getX(), points[i].getY(), points[i].getZ() };
    }

    std::vector<CLVec3> cl_refs(reference_points.size());
    for (size_t i = 0; i < reference_points.size(); ++i) {
        cl_refs[i] = { reference_points[i].getX(), reference_points[i].getY(), reference_points[i].getZ() };
    }


    // Create OpenCL buffers
    cl_mem points_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(CLVec3) * cl_points.size(), cl_points.data(), &err);
    assert(err == CL_SUCCESS);

    cl_mem refs_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(CLVec3) * cl_refs.size(), cl_refs.data(), &err);
    assert(err == CL_SUCCESS);

    cl_mem results_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        sizeof(double) * calculated_results.size(), nullptr, &err);
    assert(err == CL_SUCCESS);


    // Set kernel arguments
    int ref_count = static_cast<int>(cl_refs.size());
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &points_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &refs_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &ref_count);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &results_buf);
    assert(err == CL_SUCCESS);

    // Launch kernel
    size_t global_work_size = cl_points.size();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
    assert(err == CL_SUCCESS);

    // Read results back to calculated_results
    err = clEnqueueReadBuffer(queue, results_buf, CL_TRUE, 0,
                            sizeof(double) * calculated_results.size(),
                            calculated_results.data(), 0, nullptr, nullptr);
    assert(err == CL_SUCCESS);


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

