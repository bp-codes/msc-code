#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda

icpx -std=c++17 \
    -fsycl -O2 \
    -fsycl-targets=nvptx64-nvidia-cuda \
    sycl.cpp \
    -o sycl.x
         
        

nvcc -std=c++17 -arch=sm_86 \
        -O2 -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
        cuda.cu \
        -o cuda.x



