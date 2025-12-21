#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


g++ -O3 -std=c++17 trial_002_matrix_multiplication_serial.cpp -o trial_002_matrix_multiplication_serial.x
g++ -O3 -std=c++17 -fopenmp trial_002_matrix_multiplication_openmp.cpp -o trial_002_matrix_multiplication_openmp.x

icpx -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda -std=c++17 trial_002_matrix_multiplication_sycl.cpp -o trial_002_matrix_multiplication_sycl.x





#icpx -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda  -std=c++17 trial_001_adding_sycl2.cpp -o trial_001_adding_sycl2.x


./trial_002_matrix_multiplication_serial.x 10.0 1024
./trial_002_matrix_multiplication_openmp.x 10.0 1024 
./trial_002_matrix_multiplication_sycl.x 10.0 1024


./trial_002_matrix_multiplication_serial.x 10.0 2048
./trial_002_matrix_multiplication_openmp.x 10.0 2048 
./trial_002_matrix_multiplication_sycl.x 10.0 2048

