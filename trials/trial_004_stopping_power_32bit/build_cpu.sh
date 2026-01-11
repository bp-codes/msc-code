#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

    g++ -O3 -ffast-math -march=native -std=c++23 serial.cpp -o serial.x
    g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp openmp.cpp -o openmp.x
  
} 2>&1 | tee build.log



