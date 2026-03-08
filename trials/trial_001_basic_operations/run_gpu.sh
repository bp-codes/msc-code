#!/bin/bash

#  ./bin/sycl.x 5.0 1000000 add
#  ./bin/sycl_32.x 5.0 1000000 add
#  ./bin/cuda.x 5.0 1000000 add
#  ./bin/opencl.x 5.0 1000000 add

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
    "./bin/opencl.x"
    "./bin/opencl_32.x"
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

            if [[ "$app" == *opencl* ]]; then
                echo "    CPU"
                "$app" 5.0 1000000 "$op" CPU

                echo "    GPU"
                "$app" 5.0 1000000 "$op" GPU
            else
                "$app" 5.0 1000000 "$op"
            fi
        done

    done
done


