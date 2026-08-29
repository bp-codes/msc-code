#!/bin/bash

set -euo pipefail
export NUM_THREADS=6
mkdir -p results

RUNS=3
RUN_TIME=10.0
VECTOR_SIZE=1000000

for ((i=1; i<=RUNS; i++))
do

    # 64 bit

    ./bin/parallel_sycl.x "$RUN_TIME" "$VECTOR_SIZE" 256 CPU
    ./bin/parallel_sycl.x "$RUN_TIME" "$VECTOR_SIZE" 256 GPU
    ./bin/parallel_sycl_reduction.x "$RUN_TIME" "$VECTOR_SIZE" 256 CPU
    ./bin/parallel_sycl_reduction.x "$RUN_TIME" "$VECTOR_SIZE" 256 GPU
    ./bin/parallel_cuda.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_cuda_thrust.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_opencl.x "$RUN_TIME" "$VECTOR_SIZE" CPU
    ./bin/parallel_opencl.x "$RUN_TIME" "$VECTOR_SIZE" GPU


    # 32 bit

    ./bin/parallel_sycl_32.x 10.0 1000000 256 CPU
    ./bin/parallel_sycl_32.x 10.0 1000000 256 GPU
    ./bin/parallel_sycl_reduction_32.x "$RUN_TIME" "$VECTOR_SIZE" 256 CPU
    ./bin/parallel_sycl_reduction_32.x "$RUN_TIME" "$VECTOR_SIZE" 256 GPU
    ./bin/parallel_cuda_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_cuda_thrust_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_opencl_32.x "$RUN_TIME" "$VECTOR_SIZE" CPU
    ./bin/parallel_opencl_32.x "$RUN_TIME" "$VECTOR_SIZE" GPU


done
