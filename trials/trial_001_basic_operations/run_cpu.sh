#!/bin/bash

set -euo pipefail
export NUM_THREADS=6
mkdir -p results

RUNS=3
RUN_TIME=1.0
VECTOR_SIZE=1000000

# Precise
./bin/precise.x "$RUN_TIME" "$VECTOR_SIZE" add
./bin/precise.x "$RUN_TIME" "$VECTOR_SIZE" multiply
./bin/precise.x "$RUN_TIME" "$VECTOR_SIZE" divide
./bin/precise.x "$RUN_TIME" "$VECTOR_SIZE" power
./bin/precise.x "$RUN_TIME" "$VECTOR_SIZE" exp
./bin/precise.x "$RUN_TIME" "$VECTOR_SIZE" log
./bin/precise.x "$RUN_TIME" "$VECTOR_SIZE" sqrt

# Each entry: "executable arguments"
apps=(
    "./bin/serial_naive.x"
    "./bin/serial_naive_32.x"
    "./bin/serial_stl_transform.x"
    "./bin/serial_stl_transform_32.x"
    "./bin/serial_simd.x"
    "./bin/serial_simd_32.x"
    "./bin/parallel_stl_transform.x"
    "./bin/parallel_stl_transform_32.x"
    "./bin/parallel_thread.x"
    "./bin/parallel_thread_32.x"
    "./bin/parallel_openmp.x"
    "./bin/parallel_openmp_32.x"
    "./bin/parallel_openmp_simd.x"
    "./bin/parallel_openmp_simd_32.x"
    "./bin/precise.x"
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
