#!/bin/bash

set -euo pipefail
export NUM_THREADS=6
mkdir -p results

RUNS=5
RUN_TIME=10.0


executables=(
    ./bin/serial_naive.x
    ./bin/serial_optimized.x
    ./bin/parallel_openmp.x
    ./bin/parallel_blas.x
)

sizes=(
    "128 128 128"
    "1000 1200 800"
    "1000 1000 1000"
    "1024 1024 1024"
    "4096 4096 4096"
)

for size in "${sizes[@]}"; do
    CMD="./bin/precise.x $RUN_TIME $size"
    eval $CMD
    echo $CMD
done
echo

for ((i=1; i<=RUNS; i++))
do
    for size in "${sizes[@]}"; do
        for exe in "${executables[@]}"; do
            CMD="$exe $RUN_TIME $size"
            echo $CMD
            eval $CMD
        done
        echo
    done
done
