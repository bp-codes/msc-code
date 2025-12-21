#!/bin/bash
g++ -O2 -fopenmp main.cpp -o main.x


export OMP_NUM_THREADS=8
./main.x

