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

int main() 
{

    auto start_setup = std::chrono::high_resolution_clock::now();
    //############################################################

    const int N = 1024;
    size_t bytes = N * N * sizeof(double);
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (i + j) % 100;
            B[i * N + j] = (i * j + 3) % 100;
        }
    }


    //############################################################
    auto end_setup = std::chrono::high_resolution_clock::now();



    auto start_calc = std::chrono::high_resolution_clock::now();


    // Load kernel source
    std::string kernel_code = load_kernel(kernel_file);
    const char* source_cstr = kernel_code.c_str();
    size_t source_size = kernel_code.length();

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    err = clGetPlatformIDs(1, &platform, nullptr);
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A.data(), &err);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B.data(), &err);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);


    cl_program program = clCreateProgramWithSource(context, 1, &source_cstr, &source_size, &err);
    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2 -DCL_ENABLE_DOUBLE", nullptr, nullptr);

    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    size_t global[2] = {N, N};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr);

    double final_result = 0.0;
    for (double val : C) {
        final_result += val;
    }


    //############################################################
    auto end_calc = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> duration_setup = end_setup - start_setup;
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    std::cout << duration_setup.count() << "," << duration_calc.count() << "," << final_result << std::endl;

    // Cleanup
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
 
