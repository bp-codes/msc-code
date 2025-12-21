#!/bin/bash
nvcc -arch=sm_86 -Xcompiler -lcublas -O2 main.cu -o main.x


export OMP_NUM_THREADS=8
./main.x

