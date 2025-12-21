#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


g++ -O3 -std=c++17 trial_003_stopping_power_serial.cpp -o trial_003_stopping_power_serial.x
g++ -O3 -std=c++17 -fopenmp trial_003_stopping_power_openmp.cpp -o trial_003_stopping_power_openmp.x
icpx -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda  -std=c++17 trial_003_stopping_power_sycl_1.cpp -o trial_003_stopping_power_sycl_1.x
icpx -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda -std=c++17 trial_003_stopping_power_sycl_2.cpp -o trial_003_stopping_power_sycl_2.x
#icpx -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda  -std=c++17 trial_001_adding_sycl2.cpp -o trial_001_adding_sycl2.x


./trial_003_stopping_power_serial.x 5.0 1000000
./trial_003_stopping_power_openmp.x 5.0 1000000
./trial_003_stopping_power_sycl_1.x 5.0 1000000
./trial_003_stopping_power_sycl_2.x 5.0 1000000
#./trial_001_adding_sycl2.x

