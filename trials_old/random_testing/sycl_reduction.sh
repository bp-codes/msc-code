#!/bin/bash

acpp -std=c++17 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     sycl_reduction.cpp \
     -o sycl_reduction.x






