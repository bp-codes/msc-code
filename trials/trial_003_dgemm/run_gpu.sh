#!/bin/bash
export SYCL_DEVICE_FILTER=cuda


#./sycl.x 10.0 1000 1000

./bin/cuda.x 60.0 128 128 128
./bin/cuda_cublas.x 60.0 128 128 128



./bin/cuda.x 60.0 1000 1200 800
./bin/cuda_cublas.x 60.0 1000 1200 800



./bin/cuda.x 60.0 1000 1000 1000
./bin/cuda_cublas.x 60.0 1000 1000 1000



./bin/cuda.x 60.0 4096 4096 4096
./bin/cuda_cublas.x 60.0 4096 4096 4096






