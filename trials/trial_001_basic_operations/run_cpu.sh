#!/bin/bash

#  ./bin/precise.x 5.0 1000000 add
#  ./bin/serial.x 5.0 1000000 add

set -euo pipefail


./bin/precise.x 5.0 1000000 add
./bin/precise.x 5.0 1000000 multiply
./bin/precise.x 5.0 1000000 divide
./bin/precise.x 5.0 1000000 power
./bin/precise.x 5.0 1000000 exp
./bin/precise.x 5.0 1000000 log
./bin/precise.x 5.0 1000000 sqrt


RUNS=1
export NUM_THREADS=6

mkdir -p results

# Each entry: "executable arguments"
apps=(
    "./bin/serial.x"
    "./bin/serial_32.x"
    "./bin/serial_stl.x"
    "./bin/serial_stl_32.x"
    "./bin/serial_simd.x"
    "./bin/serial_simd_32.x"
    "./bin/parallel_stl.x"
    "./bin/parallel_stl_32.x"
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
            "$app" 5.0 1000000 "$op"
        done

    done
done


