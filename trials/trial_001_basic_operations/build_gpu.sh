#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


acpp -std=c++17 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     sycl.cpp \
     -o sycl.x





#nvcc -std=c++17 \
#    -O2 \
#    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
#    trial_001_adding_cuda.cu \
#    -o trial_001_adding_cuda.x


#g++ -std=c++17 -DCL_TARGET_OPENCL_VERSION=200 trial_001_adding_opencl_1.cpp -o trial_001_adding_opencl_1.x -lOpenCL
#g++ -std=c++17 -DCL_TARGET_OPENCL_VERSION=200 trial_001_adding_opencl_2.cpp -o trial_001_adding_opencl_2.x -lOpenCL



