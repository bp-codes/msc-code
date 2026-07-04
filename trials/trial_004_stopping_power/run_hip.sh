#!/bin/bash
set -Eeuo pipefail
export NUM_THREADS=6
mkdir -p results

RUNS=5
RUN_TIME=3.0
VECTOR_SIZE=1000000

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

    for ((i=1; i<=RUNS; i++))
    do
        echo "Run "$i
        ./bin/parallel_hip.x "$RUN_TIME" "$VECTOR_SIZE"
        ./bin/parallel_hip_32.x "$RUN_TIME" "$VECTOR_SIZE"

    done
}
