#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda

# Install headers for opencl
apt update
apt install -y pocl-opencl-icd ocl-icd-libopencl1 clinfo
mkdir -p /etc/OpenCL/vendors         
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd


# SYCL, CUDA, OpenCL 

acpp -std=c++23 -O3 -march=native -ffast-math \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl.cpp \
     -o bin/parallel_sycl.x


nvcc -std=c++20 -O3  \
    -Xcompiler "-march=native -ffast-math" \
    src/parallel_cuda.cu \
    -o bin/parallel_cuda.x


g++ -std=c++23 -O3 -march=native -ffast-math \
    src/parallel_opencl.cpp \
    -o bin/parallel_opencl.x \
    -lOpenCL



# SYCL, CUDA, OpenCL     32 bit

acpp -std=c++23 -O3 -march=native -ffast-math \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl_32.cpp \
     -o bin/parallel_sycl_32.x


nvcc -std=c++20 -O3  \
    -Xcompiler "-march=native -ffast-math" \
    src/parallel_cuda_32.cu \
    -o bin/parallel_cuda_32.x


g++ -std=c++23 -O3 -march=native -ffast-math \
    src/parallel_opencl_32.cpp \
    -o bin/parallel_opencl_32.x \
    -lOpenCL





