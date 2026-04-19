#!/bin/bash

#source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


# SYCL, CUDA, OpenCL 

acpp -std=c++23 -O3 -march=native -ffast-math \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl.cpp \
     -o bin/parallel_sycl.x

acpp -std=c++23 -O3 -march=native -ffast-math \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl_reduction.cpp \
     -o bin/parallel_sycl_reduction.x


nvcc -std=c++20 -O3  \
    -Xcompiler "-march=native -ffast-math" \
    src/parallel_cuda.cu \
    -o bin/parallel_cuda.x


nvcc -std=c++20 -O3  \
    -Xcompiler "-march=native -ffast-math" \
    src/parallel_cuda_thrust.cu \
    -o bin/parallel_cuda_thrust.x


g++ -std=c++23 -O3 -march=native -ffast-math \
    src/parallel_opencl.cpp \
    -o bin/parallel_opencl.x \
    -lOpenCL




# SYCL, CUDA, OpenCL     32 bit 

acpp -std=c++23 -O3 -march=native -ffast-math \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl_32.cpp \
     -o bin/parallel_sycl_32.x

acpp -std=c++23 -O3 -march=native -ffast-math \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl_reduction_32.cpp \
     -o bin/parallel_sycl_reduction_32.x


nvcc -std=c++20 -O3  \
    -Xcompiler "-march=native -ffast-math" \
    src/parallel_cuda_32.cu \
    -o bin/parallel_cuda_32.x


nvcc -std=c++20 -O3  \
    -Xcompiler "-march=native -ffast-math" \
    src/parallel_cuda_thrust_32.cu \
    -o bin/parallel_cuda_thrust_32.x


g++ -std=c++23 -O3 -march=native -ffast-math \
    src/parallel_opencl_32.cpp \
    -o bin/parallel_opencl_32.x \
    -lOpenCL




exit 0

acpp -O3 -ffast-math -std=c++23 \
     -v \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl.cpp \
     -o bin/parallel_sycl.x

acpp -O3 -ffast-math -std=c++23 \
     -v \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl_reduction.cpp \
     -o bin/parallel_sycl_reduction.x



nvcc -std=c++17 \
    -O3 \
     -arch=sm_86 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
       src/parallel_cuda.cu \
    -o bin/parallel_cuda.x

nvcc -std=c++17 \
    -O3 \
     -arch=sm_86 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
       src/parallel_cuda_thrust.cu \
    -o bin/parallel_cuda_thrust.x



exit 0


acpp -std=c++17 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     sycl_reduction.cpp \
     -o sycl_reduction.x


nvcc -std=c++17 \
    -O3 \
     -arch=sm_86 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
       cuda.cu \
    -o cuda.x





