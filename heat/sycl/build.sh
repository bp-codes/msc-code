#!/bin/bash

acpp -std=c++17 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     main.cpp \
     -o main.x




