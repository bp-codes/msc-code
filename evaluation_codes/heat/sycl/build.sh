#!/bin/bash
mkdir -p bin
acpp -O3 -ffast-math -std=c++23 \
     -v \
     --acpp-targets=cuda:sm_86 \
     src/main.cpp \
     -o bin/main.x




