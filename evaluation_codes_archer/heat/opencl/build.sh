#!/bin/bash
mkdir -p bin
g++ -std=c++23 -O3 -march=native -ffast-math \
    src/main.cpp \
    -o bin/heat2d.x \
    -Iinclude \
    -isystem include/nlohmann -Wno-nan-infinity-disabled \
    -lOpenCL


