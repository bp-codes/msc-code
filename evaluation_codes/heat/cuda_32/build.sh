#!/bin/bash
export SYCL_DEVICE_FILTER=cuda
mkdir -p bin
nvcc -O3 -std=c++20 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
    src/CudaEngine.cu \
    src/main.cpp \
    -o bin/main.x