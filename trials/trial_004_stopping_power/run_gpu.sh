#!/bin/bash

set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{
 
     RUNS=5
    export NUM_THREADS=6

    for ((i=1; i<=RUNS; i++))
    do

        ./bin/parallel_sycl.x 5.0 1000000 GPU
        ./bin/parallel_cuda.x 5.0 1000000
        ./bin/parallel_opencl.x 5.0 1000000 GPU
        
        ./bin/parallel_sycl_32.x 5.0 1000000 GPU
        ./bin/parallel_cuda_32.x 5.0 1000000
        ./bin/parallel_opencl_32.x 5.0 1000000 GPU

    done
}

