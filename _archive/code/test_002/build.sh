#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


cd test_002_cublas && rm main.x && ./build.sh && cd ../
cd test_002_nvcc && rm main.x && ./build.sh && cd ../
cd test_002_opencl && rm main.x && ./build.sh && cd ../
cd test_002_openmp && rm main.x && ./build.sh && cd ../
cd test_002_openmp_gpu && rm main.x && ./build.sh && cd ../
cd test_002_serial && rm main.x && ./build.sh && cd ../
cd test_002_sycl && rm main.x && ./build.sh && cd ../

