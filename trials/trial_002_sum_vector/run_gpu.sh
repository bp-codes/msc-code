#!/bin/bash

set -euo pipefail

RUNS=5

for ((i=1; i<=RUNS; i++))
do

    ./bin/parallel_sycl.x 10.0 1000000 256 cpu
    ./bin/parallel_sycl.x 10.0 1000000 256 gpu
    ./bin/parallel_sycl_32.x 10.0 1000000 256 cpu
    ./bin/parallel_sycl_32.x 10.0 1000000 256 gpu
    ./bin/parallel_sycl_reduction.x 10.0 1000000 256 cpu
    ./bin/parallel_sycl_reduction.x 10.0 1000000 256 gpu
    ./bin/parallel_cuda.x 10.0 1000000 
    ./bin/parallel_opencl.x 10.0 1000000 cpu 
    ./bin/parallel_opencl.x 10.0 1000000 gpu 


done




