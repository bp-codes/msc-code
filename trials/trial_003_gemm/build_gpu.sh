#!/bin/bash
export SYCL_DEVICE_FILTER=cuda




# SYCL, CUDA, OpenCL 

nvcc -std=c++20 -O3 \
    -arch=sm_86 -Xcompiler "-ffp-contract=fast -ffast-math" \
     src/parallel_cuda.cu \
    -o bin/parallel_cuda.x


nvcc -std=c++20 -O3 \
    -arch=sm_86 -Xcompiler "-ffp-contract=fast -ffast-math" \
     src/parallel_cuda_cublas.cu \
    -lcublas \
    -o bin/parallel_cuda_cublas.x


acpp -std=c++23 -O3 -march=native -ffast-math \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl.cpp \
     -o bin/parallel_sycl.x


acpp -std=c++23 -O3 -march=native -ffast-math \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     src/parallel_sycl_32.cpp \
     -o bin/parallel_sycl_32.x













