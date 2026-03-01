#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


g++ -O3 -std=c++17 serial.cpp -o serial.x
./serial.x
