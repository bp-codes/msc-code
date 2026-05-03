#!/bin/bash
mkdir -p bin
acpp -std=c++23 -O3 -march=native -ffast-math \
     -v \
     --acpp-targets=cuda:sm_86 \
     src/main.cpp \
     -Iinclude \
     -isystem include/nlohmann \
     -o bin/main.x




