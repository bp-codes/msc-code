#!/bin/bash

set -euo pipefail
export OMP_NUM_THREADS=32
export NUM_THREADS=32
mkdir -p results

RUNS=2
RUN_TIME=1.0
VECTOR_SIZE=1000000

# Each entry: "executable arguments"
apps=(
    "./bin/serial_naive_32.x"
    "./bin/serial_stl_transform_32.x"
    "./bin/serial_simd_32.x"
    "./bin/parallel_thread_32.x"
    "./bin/parallel_openmp_32.x"
    "./bin/parallel_openmp_simd_32.x"
)

operations=(
    "add"
    "multiply"
    "divide"
    "power"
    "exp"
    "log"
    "sqrt"
)

#./bin/serial.x 5.0 1000000 add

for app in "${apps[@]}"
do
    for op in "${operations[@]}"
    do
        echo "Running: $app ($op)"

        for ((i=1; i<=RUNS; i++))
        do
            echo "  Run $i"
            "$app" "$RUN_TIME" "$VECTOR_SIZE" "$op"
        done

        sleep 10

    done
done
