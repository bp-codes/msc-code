#!/bin/bash

g++  -O2 \
  -fno-fast-math -fno-unsafe-math-optimizations \
  -ffp-contract=off -fexcess-precision=standard \
  -std=c++17 serial.cpp \
  -o serial.x

g++  -O2 \
  -fopenmp \
  -fno-fast-math -fno-unsafe-math-optimizations \
  -ffp-contract=off -fexcess-precision=standard \
  -std=c++17 openmp.cpp \
  -o openmp.x
  


