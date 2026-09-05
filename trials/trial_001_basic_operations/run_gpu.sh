#!/bin/bash

set -euo pipefail
export NUM_THREADS=6
mkdir -p results

RUNS=2
RUN_TIME=1.0
VECTOR_SIZE=1000000

# CPU Framework executables
apps=(
    "./bin/parallel_cuda.x"
    "./bin/parallel_cuda_32.x"
    "./bin/parallel_sycl.x"
    "./bin/parallel_sycl_32.x"
    "./bin/parallel_opencl.x"
    "./bin/parallel_opencl_32.x"
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

for app in "${apps[@]}"
do
    for op in "${operations[@]}"
    do
        echo "Running: $app ($op)"

        for ((i=1; i<=RUNS; i++))
        do
            echo "  Run $i"

            if [[ "$app" == *opencl* || "$app" == *sycl* ]]; then

                echo "    CPU"
                CMD="$app "$RUN_TIME" "$VECTOR_SIZE" $op CPU"
                echo $CMD
                eval $CMD

                echo "    GPU"
                CMD="$app "$RUN_TIME" "$VECTOR_SIZE" $op GPU"
                echo $CMD
                eval $CMD

            else

                CMD="$app "$RUN_TIME" "$VECTOR_SIZE" $op"
                echo $CMD
                eval $CMD

            fi
        done

    done
done
