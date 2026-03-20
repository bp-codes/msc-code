#!/bin/bash

set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{
 
    ./bin/parallel_sycl.x 5.0 1000000 GPU
    ./bin/parallel_sycl_32.x 5.0 1000000 GPU
    ./bin/parallel_cuda.x 5.0 1000000
    ./bin/parallel_cuda_32.x 5.0 1000000

}

