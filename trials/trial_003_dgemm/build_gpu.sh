#!/bin/bash
export SYCL_DEVICE_FILTER=cuda


nvcc -std=c++20 -O3 \
    -arch=sm_86 -Xcompiler "-ffp-contract=fast -ffast-math" \
     src/cuda.cu \
    -o bin/cuda.x



nvcc -std=c++20 -O3 \
    -arch=sm_86 -Xcompiler "-ffp-contract=fast -ffast-math" \
     src/cuda_cublas.cu \
    -lcublas \
    -o bin/cuda_cublas.x

