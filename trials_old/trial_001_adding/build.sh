#!/bin/bash

g++ -O3 -std=c++17 trial_001_adding_serial.cpp -o trial_001_adding_serial.x
acpp -v --acpp-targets=cuda:sm_86 trial_001_adding_sycl_1.cpp -O3 -o trial_001_adding_sycl_1.x
acpp -v --acpp-targets=cuda:sm_86 trial_001_adding_sycl_2.cpp -O3 -o trial_001_adding_sycl_2.x
nvcc -O3 trial_001_adding_cuda.cu -o trial_001_adding_cuda.x


./trial_001_adding_serial.x 5.0 10000000
./trial_001_adding_sycl_1.x 5.0 10000000
./trial_001_adding_sycl_2.x 5.0 10000000
./trial_001_adding_cuda.x 5.0 10000000


exit 0

g++ -O3 -std=c++17 trial_001_adding_serial.cpp -o trial_001_adding_serial.x
g++ -O3 -std=c++17 -fopenmp trial_001_adding_openmp.cpp -o trial_001_adding_openmp.x
icpx -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda -std=c++17 trial_001_adding_sycl_1.cpp -o trial_001_adding_sycl_1.x
icpx -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda -std=c++17 trial_001_adding_sycl_2.cpp -o trial_001_adding_sycl_2.x
nvcc -O3 trial_001_adding_cuda.cu -o trial_001_adding_cuda.x
g++ -std=c++17 -DCL_TARGET_OPENCL_VERSION=200 trial_001_adding_opencl_1.cpp -o trial_001_adding_opencl_1.x -lOpenCL
g++ -std=c++17 -DCL_TARGET_OPENCL_VERSION=200 trial_001_adding_opencl_2.cpp -o trial_001_adding_opencl_2.x -lOpenCL


./trial_001_adding_serial.x 5.0 10000000
./trial_001_adding_openmp.x 5.0 10000000
./trial_001_adding_sycl_1.x 5.0 10000000
./trial_001_adding_sycl_2.x 5.0 10000000
./trial_001_adding_cuda.x 5.0 10000000
./trial_001_adding_opencl_1.x 5.0 10000000
./trial_001_adding_opencl_2.x 5.0 10000000
