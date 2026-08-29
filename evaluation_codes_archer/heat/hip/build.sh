#!/bin/bash
mkdir -p bin

# HIP
hipcc -std=c++20 -O3 \
     --gpu-architecture=sm_86 \
     src/main.cpp \
     -Iinclude \
     -isystem include/nlohmann \
     -o bin/main.x


