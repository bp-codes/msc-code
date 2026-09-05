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
    "./bin/parallel_hip.x"
    "./bin/parallel_hip_32.x"
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

                CMD="$app "$RUN_TIME" "$VECTOR_SIZE" $op"
                echo $CMD
                eval $CMD


        done

    done
done
