#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda
icpx -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda  -std=c++17 sycl_test.cpp -o sycl_test.x
./sycl_test.x

