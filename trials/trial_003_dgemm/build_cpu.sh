#!/bin/bash

mkdir -p bin

g++ -std=c++23 -O3 -march=native -ffast-math \
   src/serial_naive.cpp \
  -o bin/serial_naive.x -lopenblas

g++ -std=c++23 -O3 -ffast-math  \
  -mavx2 -mfma \
   src/serial_optimized.cpp \
  -o bin/serial_optimized.x -lopenblas -fopenmp





g++ -std=c++23 -O3 -ffp-contract=fast -ffast-math \
  -mavx2 -mfma \
   src/parallel_blas.cpp \
  -o bin/parallel_blas.x -lopenblas

g++ -std=c++23 -O3 -ffp-contract=fast -ffast-math \
  -mavx2 -mfma \
   src/parallel_blas_32.cpp \
  -o bin/parallel_blas_32.x -lopenblas

g++ -std=c++23 -O3 -ffp-contract=fast -ffast-math \
  -mavx2 -mfma \
   src/parallel_openmp.cpp \
  -o bin/parallel_openmp.x -lopenblas -fopenmp






