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

exit 0

acpp -std=c++23 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     sycl1.cpp \
     -o sycl1.x




