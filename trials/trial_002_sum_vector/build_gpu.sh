#!/bin/bash

acpp -std=c++17 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     sycl.cpp \
     -o sycl.x


acpp -std=c++17 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     sycl_reduction.cpp \
     -o sycl_reduction.x


nvcc -std=c++17 \
    -O3 \
     -arch=sm_86 \
    -Xcompiler "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off" \
       cuda.cu \
    -o cuda.x





