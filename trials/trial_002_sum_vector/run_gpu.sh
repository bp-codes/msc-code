#!/bin/bash

apt update && apt install time
set -euo pipefail

RUNS=5

./bin/parallel_sycl.x 10.0 1000000 256 gpu
./bin/parallel_sycl_reduction.x 10.0 1000000 256 gpu
./bin/parallel_cuda.x 10.0 1000000 
#./sycl_reduction.x 10.0 10000000 256 gpu
#./cuda.x 10.0 10000000




