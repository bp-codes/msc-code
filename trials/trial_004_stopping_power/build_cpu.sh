#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

    g++ -O3 -ffast-math -march=native -std=c++23 -fext-numeric-literals src/precise.cpp -lquadmath -o bin/precise.x

    g++ -O3 -ffast-math -march=native -std=c++23 src/serial.cpp -o bin/serial.x
    g++ -O3 -ffast-math -march=native -std=c++23 src/parallel_transform.cpp -ltbb -o bin/parallel_transform.x
    g++ -O3 -ffast-math -march=native -std=c++23 src/parallel_thread.cpp -o bin/parallel_thread.x
    g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp.cpp -o bin/parallel_openmp.x
    g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp_simd.cpp -o bin/parallel_openmp_simd.x


    g++ -O3 -ffast-math -march=native -std=c++23 src/serial_32.cpp -o bin/serial_32.x
    g++ -O3 -ffast-math -march=native -std=c++23 src/parallel_transform_32.cpp -ltbb -o bin/parallel_transform_32.x
    g++ -O3 -ffast-math -march=native -std=c++23 src/parallel_thread_32.cpp -o bin/parallel_thread_32.x
    g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp_32.cpp -o bin/parallel_openmp_32.x
    g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp_simd_32.cpp -o bin/parallel_openmp_simd_32.x
  
} 2>&1 | tee build.log


