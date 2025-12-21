#!/bin/bash
g++ main.cpp -o main.x -lOpenCL


export OMP_NUM_THREADS=8
./main.x

