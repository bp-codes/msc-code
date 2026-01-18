#!/bin/bash

acpp -O3 -ffast-math -std=c++23 \
     -v \
     --acpp-targets=cuda:sm_86 \
     main.cpp \
     -o main.x




