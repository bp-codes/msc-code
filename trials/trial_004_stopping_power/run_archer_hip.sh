#!/bin/bash
set -Eeuo pipefail
export OMP_NUM_THREADS=32
export NUM_THREADS=32
mkdir -p results

RUNS=5
RUN_TIME=3.0
VECTOR_SIZE=1000000

TIME="/usr/bin/time -v --"
TIME=""

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{
    ./bin/precise.x "$RUN_TIME" "$VECTOR_SIZE"

    for ((i=1; i<=RUNS; i++))
    do
        echo "  Run $i"

        $TIME ./bin/parallel_hip.x "$RUN_TIME" "$VECTOR_SIZE"


        $TIME ./bin/parallel_hip_32.x "$RUN_TIME" "$VECTOR_SIZE"

    done

} 2>&1 | tee results_archer_hip.log
