#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{

    mkdir -p bin

    g++ -O3 -ffast-math -march=native -std=c++23 src/serial.cpp -o bin/serial.x
    g++ -O3 -ffast-math -march=native -std=c++23 src/serial_stl.cpp -o bin/serial_stl.x
    g++ -O3 -ffast-math -march=native -std=c++23 -mavx2 -mfma src/serial_simd.cpp -o bin/serial_simd.x
    g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/openmp.cpp -o bin/openmp.x
    g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/openmp_simd.cpp -o bin/openmp_simd.x
  
} 2>&1 | tee build.log









