#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{
    module restore
    module load PrgEnv-gnu

    mkdir -p bin

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

    # CPU Parallel
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


    # CPU Serial
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

    # CPU Parallel
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
