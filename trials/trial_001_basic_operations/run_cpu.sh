#!/bin/bash

set -euo pipefail

RUNS=1
export OMP_NUM_THREADS=4

mkdir -p results

# Each entry: "executable arguments"
apps=(
    "./bin/serial.x"
    "./bin/serial32.x"
    "./bin/serial_simd.x"
    "./bin/serial_simd_32.x"
    "./bin/serial_stl.x"
    "./bin/serial_stl_32.x"
    "./bin/openmp.x"
    "./bin/openmp_32.x"
    "./bin/openmp_simd.x"
    "./bin/openmp_simd_32.x"
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


