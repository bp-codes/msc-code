#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

    g++ -O3 -ffast-math -march=native -std=c++23 src/serial.cpp -o bin/serial.x
    g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp.cpp -o bin/parallel_openmp.x
    g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp_simd.cpp -o bin/parallel_openmp_simd.x
  
} 2>&1 | tee build.log


