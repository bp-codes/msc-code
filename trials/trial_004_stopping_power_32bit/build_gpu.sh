#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


acpp -std=c++23 \
     -O3 \
     -v \
     -fopenmp \
     --acpp-targets=cuda:sm_86 \
     sycl.cpp \
     -o sycl.x

exit 0



