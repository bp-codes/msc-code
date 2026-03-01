#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


acpp -std=c++23 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     sycl.cpp \
     -o sycl.x

nvcc -std=c++17 \
    -O3 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
    cuda.cu \
    -o cuda.x

exit 0

acpp -std=c++23 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     sycl1.cpp \
     -o sycl1.x




