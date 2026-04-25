#!/bin/bash
set -Eeuo pipefail
export NUM_THREADS=6
mkdir -p results

RUNS=2
RUN_TIME=2.0
VECTOR_SIZE=1000000

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

     RUNS=5
    export NUM_THREADS=6

    for ((i=1; i<=RUNS; i++))
    do

        ./bin/parallel_sycl.x "$RUN_TIME" "$VECTOR_SIZE" GPU
        ./bin/parallel_sycl.x "$RUN_TIME" "$VECTOR_SIZE" CPU
        ./bin/parallel_cuda.x "$RUN_TIME" "$VECTOR_SIZE"
        ./bin/parallel_opencl.x "$RUN_TIME" "$VECTOR_SIZE" GPU
        ./bin/parallel_opencl.x "$RUN_TIME" "$VECTOR_SIZE" CPU

        ./bin/parallel_sycl_32.x "$RUN_TIME" "$VECTOR_SIZE" GPU
        ./bin/parallel_sycl_32.x "$RUN_TIME" "$VECTOR_SIZE" CPU
        ./bin/parallel_cuda_32.x "$RUN_TIME" "$VECTOR_SIZE"
        ./bin/parallel_opencl_32.x "$RUN_TIME" "$VECTOR_SIZE" GPU
        ./bin/parallel_opencl_32.x "$RUN_TIME" "$VECTOR_SIZE" CPU

    done
}
