#!/bin/bash
mkdir -p bin
g++ -std=c++23 -O3 -march=native -ffast-math \
    src/main.cpp \
    -Iinclude \
    -isystem include/nlohmann \
    -o bin/main.x

