#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{

    mkdir -p bin

    # CPU Precise
    g++ -std=c++23 -O3 -march=native -ffast-math \
        -fext-numeric-literals \
         src/precise.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/precise.x \
        -lquadmath

    # CPU Serial
    g++ -std=c++23 -O3 -march=native -ffast-math \
         src/serial_naive.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/serial_naive.x
    g++ -std=c++23 -O3 -march=native -ffast-math \
         src/serial_stl_transform.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/serial_stl_transform.x
    g++ -std=c++23 -O3 -march=native -ffast-math \
        -mavx2 -mfma \
         src/serial_simd.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/serial_simd.x

    # CPU Parallel
    g++ -std=c++23 -O3 -march=native -ffast-math \
         src/parallel_stl_transform.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_stl_transform.x \
        -ltbb
    g++ -std=c++23 -O3 -march=native -ffast-math \
         src/parallel_thread.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_thread.x
    g++ -std=c++23 -O3 -march=native -ffast-math \
        -fopenmp \
         src/parallel_openmp.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_openmp.x
    g++ -std=c++23 -O3 -march=native -ffast-math \
        -fopenmp \
         src/parallel_openmp_simd.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_openmp_simd.x

    # CPU Serial 32 bit
    g++ -std=c++23 -O3 -march=native -ffast-math \
         src/serial_naive_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/serial_naive_32.x
    g++ -std=c++23 -O3 -march=native -ffast-math \
         src/serial_stl_transform_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/serial_stl_transform_32.x
    g++ -std=c++23 -O3 -march=native -ffast-math \
        -mavx2 -mfma \
         src/serial_simd_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/serial_simd_32.x

    # CPU Parallel 32 bit
    g++ -std=c++23 -O3 -march=native -ffast-math \
         src/parallel_stl_transform_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_stl_transform_32.x \
        -ltbb
    g++ -std=c++23 -O3 -march=native -ffast-math \
         src/parallel_thread_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_thread_32.x
    g++ -std=c++23 -O3 -march=native -ffast-math \
        -fopenmp \
         src/parallel_openmp_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_openmp_32.x
    g++ -std=c++23 -O3 -march=native -ffast-math \
        -fopenmp \
         src/parallel_openmp_simd_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_openmp_simd_32.x



} 2>&1 | tee build.log
