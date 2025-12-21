#!/bin/bash
export OMP_NUM_THREADS=8
g++ -O2 -fopenmp main.cpp -o main.x
./main.x


