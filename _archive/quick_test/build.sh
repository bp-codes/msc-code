#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda
g++ -O3 -std=c++17 serial_stress.cpp -o serial_stress.x
g++ -O3 -fopenmp openmp.cpp -o openmp.x
icpx -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda  -std=c++17 sycl.cpp -o sycl.x

./serial_stress.x
./openmp.x
./sycl.x

