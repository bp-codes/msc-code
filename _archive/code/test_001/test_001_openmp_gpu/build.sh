#!/bin/bash
g++-13 -fopenmp -foffload=nvptx-none \
  -fcf-protection=none -fno-stack-protector -no-pie \
  -foffload-options=nvptx-none="-fcf-protection=none -lm" \
  -O2 -std=c++17 main.cpp -o main.x -lm

export OMP_NUM_THREADS=8
./main.x

