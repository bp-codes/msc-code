#!/bin/bash
mkdir -p bin
g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/main.cpp -o bin/main.x
