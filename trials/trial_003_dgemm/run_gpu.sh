#!/bin/bash
export SYCL_DEVICE_FILTER=cuda


#./sycl.x 10.0 1000 1000

./bin/parallel_cuda.x 60.0 128 128 128
./bin/parallel_cuda_cublas.x 60.0 128 128 128
./bin/parallel_sycl.x 60.0 128 128 128
./bin/parallel_sycl_32.x 60.0 128 128 128


./bin/parallel_cuda.x 60.0 1000 1200 800
./bin/parallel_cuda_cublas.x 60.0 1000 1200 800


./bin/parallel_cuda.x 60.0 1000 1000 1000
./bin/parallel_cuda_cublas.x 60.0 1000 1000 1000


./bin/parallel_cuda.x 60.0 1024 1024 1024
./bin/parallel_cuda_cublas.x 60.0 1024 1024 1024


./bin/parallel_cuda.x 60.0 4096 4096 4096
./bin/parallel_cuda_cublas.x 60.0 4096 4096 4096
./bin/parallel_sycl.x 60.0 4096 4096 4096
./bin/parallel_sycl_32.x 60.0 4096 4096 4096






