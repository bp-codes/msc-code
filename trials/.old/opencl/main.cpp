// g++ -std=c++20 main.cpp -lOpenCL -O2

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cstdlib>

const char* kernelSource = R"CLC(
__kernel void vector_add(__global const float* A,
                         __global const float* B,
                         __global float* C)
{
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}
)CLC";

int main()
{
    const std::size_t N = 1024;
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    cl_int err;

    // Platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // Device (GPU preferred)
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // Context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    // Command queue
    cl_command_queue queue =
        clCreateCommandQueue(context, device, 0, &err);

    // Buffers
    cl_mem bufferA =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float)*N, A.data(), &err);

    cl_mem bufferB =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float)*N, B.data(), &err);

    cl_mem bufferC =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                       sizeof(float)*N, nullptr, &err);

    // Program
    const char* source = kernelSource;
    cl_program program =
        clCreateProgramWithSource(context, 1, &source, nullptr, &err);

    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    // Kernel
    cl_kernel kernel =
        clCreateKernel(program, "vector_add", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    // Launch
    size_t globalSize = N;
    clEnqueueNDRangeKernel(queue, kernel, 1,
                           nullptr, &globalSize, nullptr,
                           0, nullptr, nullptr);

    // Read back
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0,
                        sizeof(float)*N, C.data(),
                        0, nullptr, nullptr);

    // Verify
    for (int i = 0; i < 5; ++i)
        std::cout << C[i] << "\n";

    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
} 
