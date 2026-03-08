#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


acpp -O3 -ffast-math -std=c++23 \
     -v \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl.cpp \
     -o bin/parallel_sycl.x


acpp -O3 -ffast-math -std=c++23 \
     -v \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl_32.cpp \
     -o bin/parallel_sycl_32.x


nvcc -std=c++17 \
    -O3 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
    src/parallel_cuda.cu \
    -o bin/parallel_cuda.x


nvcc -std=c++17 \
    -O2 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
    src/parallel_cuda_32.cu \
    -o bin/parallel_cuda_32.x


g++ -std=c++23 \
    -O3 \
    src/parallel_opencl.cpp \
    -o bin/parallel_opencl.x \
    -lOpenCL


g++ -std=c++23 \
    -O3 \
    src/parallel_opencl_32.cpp \
    -o bin/parallel_opencl_32.x \
    -lOpenCL


#g++ -std=c++17 -DCL_TARGET_OPENCL_VERSION=200 trial_001_adding_opencl_1.cpp -o trial_001_adding_opencl_1.x -lOpenCL
#g++ -std=c++17 -DCL_TARGET_OPENCL_VERSION=200 trial_001_adding_opencl_2.cpp -o trial_001_adding_opencl_2.x -lOpenCL



