#!/bin/bash
nvcc -Xcompiler -fopenmp -lcublas -O3 -o main.x main.cu


export OMP_NUM_THREADS=8
./main.x

