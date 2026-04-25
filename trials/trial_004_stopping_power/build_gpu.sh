#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda

# apt update
# apt install -y ocl-icd-opencl-dev
#


acpp -O3 -ffast-math -std=c++23 \
     -v \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl.cpp \
    -Iinclude \
    -isystem include/nlohmann \
     -o bin/parallel_sycl.x


acpp -O3 -ffast-math -std=c++23 \
     -v \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl_32.cpp \
    -Iinclude \
    -isystem include/nlohmann \
     -o bin/parallel_sycl_32.x


nvcc -std=c++20 --use_fast_math \
    -O3 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
    src/parallel_cuda.cu \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/parallel_cuda.x

nvcc -std=c++20 --use_fast_math \
    -O3 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
    src/parallel_cuda_32.cu \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/parallel_cuda_32.x


g++ -std=c++23 \
    -O3 \
    src/parallel_opencl.cpp \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/parallel_opencl.x \
    -lOpenCL


g++ -std=c++23 \
    -O3 \
    src/parallel_opencl_32.cpp \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/parallel_opencl_32.x \
    -lOpenCL


exit 0

acpp -std=c++23 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     sycl1.cpp \
     -o sycl1.x
