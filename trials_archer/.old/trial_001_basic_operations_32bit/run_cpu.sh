#!/bin/bash

set -euo pipefail

RUNS=5
OUTFILE="benchmark_results_1.json"
export OMP_NUM_THREADS=4

mkdir -p results

# Each entry: "executable arguments"
apps=(
    "./bin/serial.x 5.0 1000000 add"
    "./bin/serial_simd.x 5.0 1000000 add"
    "./bin/serial_stl.x 5.0 1000000 add"
    "./bin/openmp.x 5.0 1000000 add"
    "./bin/openmp_simd.x 5.0 1000000 add"
)

./bin/serial.x 5.0 1000000 add




