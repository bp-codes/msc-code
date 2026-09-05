#!/bin/bash
export SYCL_DEVICE_FILTER=cuda




# SYCL, CUDA, OpenCL

nvcc -std=c++20 -O3 \
    -arch=sm_86 -Xcompiler "-ffp-contract=fast -ffast-math" \
     src/parallel_cuda.cu \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/parallel_cuda.x

nvcc -std=c++20 -O3 \
    -arch=sm_86 -Xcompiler "-ffp-contract=fast -ffast-math" \
     src/parallel_cuda_cublas.cu \
    -lcublas \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/parallel_cuda_cublas.x

acpp -std=c++23 -O3 -march=native -ffast-math \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl.cpp \
    -Iinclude \
    -isystem include/nlohmann \
     -o bin/parallel_sycl.x

g++ -std=c++23 -O3 -march=native -ffast-math \
    src/parallel_opencl.cpp \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/parallel_opencl.x \
    -lOpenCL



# SYCL, CUDA, OpenCL  32 bit

nvcc -std=c++20 -O3 \
    -arch=sm_86 -Xcompiler "-ffp-contract=fast -ffast-math" \
     src/parallel_cuda_32.cu \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/parallel_cuda_32.x

nvcc -std=c++20 -O3 \
    -arch=sm_86 -Xcompiler "-ffp-contract=fast -ffast-math" \
     src/parallel_cuda_cublas_32.cu \
    -lcublas \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/parallel_cuda_cublas_32.x

acpp -std=c++23 -O3 -march=native -ffast-math \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl_32.cpp \
    -Iinclude \
    -isystem include/nlohmann \
     -o bin/parallel_sycl_32.x

g++ -std=c++23 -O3 -march=native -ffast-math \
    src/parallel_opencl_32.cpp \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/parallel_opencl_32.x \
    -lOpenCL
