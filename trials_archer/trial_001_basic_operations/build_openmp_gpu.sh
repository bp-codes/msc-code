#!/bin/bash

# Install headers for opencl
#apt update
#apt install -y pocl-opencl-icd ocl-icd-libopencl1 clinfo
#mkdir -p /etc/OpenCL/vendors
#echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd


# OpenMP GPU

g++-13 -O3 -std=c++20 -fopenmp \
    -fno-stack-protector \
    -fcf-protection=none \
    -fno-math-errno \
    -foffload=nvptx-none \
    -foffload-options=nvptx-none="-misa=sm_80 -fno-math-errno -lm" \
     src/parallel_openmp_gpu_offload.cpp \
    -Iinclude \
    -isystem include/nlohmann -Wno-nan-infinity-disabled \
     -o bin/parallel_openmp_gpu_offload.x


g++-13 -O3 -std=c++20 -fopenmp \
    -fno-stack-protector \
    -fcf-protection=none \
    -fno-math-errno \
    -foffload=nvptx-none \
    -foffload-options=nvptx-none="-misa=sm_80 -fno-math-errno -lm" \
     src/parallel_openmp_gpu_offload_32.cpp \
    -Iinclude \
    -isystem include/nlohmann -Wno-nan-infinity-disabled \
     -o bin/parallel_openmp_gpu_offload_32.x
