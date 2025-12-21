#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

const char* kernel_file = "kernel.cl";

std::string load_kernel(const char* filename) {
    std::ifstream file(filename);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

int main() {
    constexpr int num_iterations = 100'000'000;
    std::vector<double> results(num_iterations);


    auto start_setup = std::chrono::high_resolution_clock::now();
    //############################################################



    // Load kernel source
    std::string kernel_code = load_kernel(kernel_file);
    const char* source = kernel_code.c_str();

    cl_int err;
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_platform_id platform = platforms[0];

    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

    cl_device_id device = devices[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    cl_mem result_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * num_iterations, nullptr, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -DCL_KHR_FP64", nullptr, nullptr);

    if (err != CL_SUCCESS) {
        char build_log[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, nullptr);
        std::cerr << "Build Error:\n" << build_log << "\n";
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "exp_kernel", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &result_buf);
    clSetKernelArg(kernel, 1, sizeof(int), &num_iterations);

    size_t global_work_size = num_iterations;


    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    

    clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0, sizeof(double) * num_iterations, results.data(), 0, nullptr, nullptr);


    //############################################################
    auto end_setup = std::chrono::high_resolution_clock::now();



    auto start_calc = std::chrono::high_resolution_clock::now();

    double final_result = 0.0;
    for (int i = 0; i < num_iterations; ++i) {
        final_result += results[i];
    }


    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << final_result << std::endl;

    // Cleanup
    clReleaseMemObject(result_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
 
