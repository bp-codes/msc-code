#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda
icpx -fsycl -O2 -std=c++17 main.cpp -o main.x
./main.x

