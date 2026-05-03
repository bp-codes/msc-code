#!/bin/bash

set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{

    mkdir -p bin

    # CPU Precise
    g++ -std=c++23 -O3 -march=native -ffast-math \
        src/precise.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/precise.x -lquadmath


    # CPU Serial
    g++ -std=c++23 -O3 -march=native -ffast-math \
      src/serial_naive.cpp \
        -Iinclude \
        -isystem include/nlohmann \
      -o bin/serial_naive.x -lopenblas
    g++ -std=c++23 -O3 -ffast-math  \
      -mavx2 -mfma \
      src/serial_optimized.cpp \
        -Iinclude \
        -isystem include/nlohmann \
      -o bin/serial_optimized.x -lopenblas -fopenmp


    # CPU Parallel
    g++ -std=c++23 -O3 -ffp-contract=fast -ffast-math \
        -mavx2 -mfma \
        src/parallel_blas.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_blas.x -lopenblas
    g++ -std=c++23 -O3 -ffp-contract=fast -ffast-math \
        -mavx2 -mfma \
        src/parallel_openmp.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_openmp.x -lopenblas -fopenmp



    # CPU Serial 32 bit
    g++ -std=c++23 -O3 -march=native -ffast-math \
        src/serial_naive_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/serial_naive_32.x -lopenblas
    g++ -std=c++23 -O3 -ffast-math  \
        -mavx2 -mfma \
        src/serial_optimized_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/serial_optimized_32.x -lopenblas -fopenmp


    # CPU Parallel 32 bit
    g++ -std=c++23 -O3 -ffp-contract=fast -ffast-math \
        -mavx2 -mfma \
        src/parallel_blas_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_blas_32.x -lopenblas
    g++ -std=c++23 -O3 -ffp-contract=fast -ffast-math \
        -mavx2 -mfma \
        src/parallel_openmp_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_openmp_32.x -lopenblas -fopenmp



} 2>&1 | tee build.log
