#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

    g++ -O3 -ffast-math -march=native -std=c++17 serial.cpp -o serial.x
    g++ -O3 -ffast-math -march=native -std=c++17 serial_stl.cpp -o serial_stl.x
    g++ -O3 -ffast-math -march=native -std=c++17 -mavx2 -mfma serial_simd.cpp -o serial_simd.x
    g++ -O3 -ffast-math -march=native -std=c++17 -fopenmp openmp.cpp -o openmp.x
    g++ -O3 -ffast-math -march=native -std=c++17 -fopenmp openmp_simd.cpp -o openmp_simd.x
  
} 2>&1 | tee build.log









