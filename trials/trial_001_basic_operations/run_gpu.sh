#!/bin/bash

apt update && apt install time
set -euo pipefail

RUNS=5

mkdir -p results

# Each entry: "executable arguments"
apps=(
    "./bin/cuda.x"
    "./bin/cuda_32.x"
    "./bin/sycl.x"
    "./bin/sycl_32.x"
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


