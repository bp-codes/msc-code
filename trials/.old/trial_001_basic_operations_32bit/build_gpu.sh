#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


acpp -O3 -ffast-math -std=c++23 \
     -v \
     --acpp-targets=cuda:sm_86 \
     src/sycl.cpp \
     -o bin/sycl.x





nvcc -std=c++17 \
    -O2 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
    src/cuda.cu \
    -o bin/cuda.x


#g++ -std=c++17 -DCL_TARGET_OPENCL_VERSION=200 trial_001_adding_opencl_1.cpp -o trial_001_adding_opencl_1.x -lOpenCL
#g++ -std=c++17 -DCL_TARGET_OPENCL_VERSION=200 trial_001_adding_opencl_2.cpp -o trial_001_adding_opencl_2.x -lOpenCL



