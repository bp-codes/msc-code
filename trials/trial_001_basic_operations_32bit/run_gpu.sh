#!/bin/bash

apt update && apt install time

set -euo pipefail

RUNS=5
OUTFILE="benchmark_results_gpu.json"
export OMP_NUM_THREADS=4

# Each entry: "executable arguments"
apps=(
    "./bin/sycl.x 5.0 1000000 add"
)

mkdir -p results

./bin/sycl.x 5.0 1000000 add



