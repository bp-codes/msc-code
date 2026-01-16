#!/bin/bash
export SYCL_DEVICE_FILTER=cuda

nvcc -O3 -std=c++20 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
    CudaEngine.cu \
    main.cpp \
    -o main.x