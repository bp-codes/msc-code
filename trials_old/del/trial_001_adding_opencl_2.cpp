// trial_001_adding_opencl.cpp
// Build: g++ -std=c++17 -DCL_TARGET_OPENCL_VERSION=200 trial_001_adding_opencl.cpp -o trial_001_adding_opencl.x -lOpenCL
#include <CL/cl.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cstring>

static inline std::size_t round_up(std::size_t n, std::size_t m) {
    return ((n + m - 1) / m) * m;
}

static void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::cerr << what << " failed with error " << err << std::endl;
        std::exit(2);
    }
}

// OpenCL kernel: each work-item accumulates a private sum over a strided slice,
// then atomically adds it to the global result.
static const char* kSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void sum_kernel(__global const int* data,
                         const int n,
                         __global int* out_sum)
{
    int gid    = get_global_id(0);
    int stride = get_global_size(0);

    int local_sum = 0;
    for (int i = gid; i < n; i += stride) {
        local_sum += data[i];
    }

#if __OPENCL_C_VERSION__ >= 200
    atomic_fetch_add((_Atomic int*)out_sum, local_sum);
#else
    atomic_add(out_sum, local_sum);
#endif
}
)CLC";

int main(int argc, char** argv) {

    // Load input options (run time, vector size)
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " time_limit  vec_size\n";
        return 1;
    }

    double test_time = std::atof(argv[1]);
    const int N = std::atoi(argv[2]);

    // Fill vector with integers
    std::vector<int> numbers;
    numbers.reserve(N);
    int last{};
    for (int i = 0; i < N; ++i) {
        int n = 8039 * (last + i + 550607) % 10000;
        numbers.push_back(n);
        last = n;
    }

    // OpenCL setup
    cl_int err = CL_SUCCESS;

    cl_uint numPlatforms = 0;
    check(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs(count)");
    if (!numPlatforms) 
    { 
        std::cerr << "No OpenCL platforms found\n"; 
        return 2; 
    }

    std::vector<cl_platform_id> plats(numPlatforms);
    check(clGetPlatformIDs(numPlatforms, plats.data(), nullptr), "clGetPlatformIDs");

    cl_platform_id plat = plats[0];

    cl_uint numDevices = 0;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
    if (err == CL_DEVICE_NOT_FOUND || numDevices == 0) 
    {
        std::cerr << "No OpenCL devices on the first platform\n"; return 2;
    }
    check(err, "clGetDeviceIDs(count)");
    std::vector<cl_device_id> devs(numDevices);
    check(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, numDevices, devs.data(), nullptr), "clGetDeviceIDs");

    cl_device_id dev = devs[0];

    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    check(err, "clCreateContext");

    // Create a command queue (profiling disabled is fine for this)
    #if defined(CL_VERSION_2_0)
        cl_command_queue opencl_queue = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);
    #else
        cl_command_queue opencl_queue = clCreateCommandQueue(ctx, dev, 0, &err);
    #endif
    check(err, "clCreateCommandQueue");

    // --- Build program & kernel ---
    const char* sources[] = { kSource };
    size_t lengths[] = { std::strlen(kSource) };
    cl_program prog = clCreateProgramWithSource(ctx, 1, sources, lengths, &err);
    check(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Print build log for easier debugging
        size_t logSize = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cerr << "Build log:\n" << log << std::endl;
        check(err, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(prog, "sum_kernel", &err);
    check(err, "clCreateKernel");

    // --- Buffers (reused across iterations) ---
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(int) * numbers.size(), numbers.data(), &err);
    check(err, "clCreateBuffer(d_in)");

    cl_mem d_sum = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &err);
    check(err, "clCreateBuffer(d_sum)");

    // --- Kernel launch configuration ---
    const size_t local = 256;
    // Use a decent global size; striding in the kernel means it doesn't have to match N
    size_t global = round_up( std::min<std::size_t>(static_cast<std::size_t>(N), 1<<20), local );
    if (global == 0) global = local; // handle N==0 gracefully

    // Timer
    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(test_time);
    std::uint64_t iters = {};
    int sum {};

    // Run as many iterations as possible within the time limit
    do 
    {
        // Zero the device-side sum
        const cl_int zero = 0;
        check(clEnqueueWriteBuffer(opencl_queue, d_sum, CL_FALSE, 0, sizeof(cl_int), &zero, 0, nullptr, nullptr),
              "clEnqueueWriteBuffer(d_sum=0)");

        // Set args: (__global const int* data, int n, __global int* out_sum)
        check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in),  "clSetKernelArg(0)");
        check(clSetKernelArg(kernel, 1, sizeof(cl_int), &N),     "clSetKernelArg(1)");
        check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_sum), "clSetKernelArg(2)");

        // Launch
        check(clEnqueueNDRangeKernel(opencl_queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
              "clEnqueueNDRangeKernel");

        // Read back result (blocking read ensures kernel finished)
        check(clEnqueueReadBuffer(opencl_queue, d_sum, CL_TRUE, 0, sizeof(cl_int), &sum, 0, nullptr, nullptr),
              "clEnqueueReadBuffer(d_sum)");

        ++iters;
    } while (std::chrono::steady_clock::now() < deadline);

    std::cout << "OpenCL2," << iters << "," << sum << std::endl;

    // --- Cleanup ---
    clReleaseMemObject(d_sum);
    clReleaseMemObject(d_in);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(opencl_queue);
    clReleaseContext(ctx);

    return 0;
}
