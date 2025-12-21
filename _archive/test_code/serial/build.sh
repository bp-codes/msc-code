#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda
g++ -O3 -std=c++17 serial_stress.cpp -o serial_stress.x
./serial_stress.x

