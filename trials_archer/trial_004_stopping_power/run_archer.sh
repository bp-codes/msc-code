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

        $TIME ./bin/serial_naive.x "$RUN_TIME" "$VECTOR_SIZE"
        #$TIME ./bin/serial_stl_transform.x "$RUN_TIME" "$VECTOR_SIZE"
        #$TIME ./bin/parallel_stl_transform.x "$RUN_TIME" "$VECTOR_SIZE"
        $TIME ./bin/parallel_thread.x "$RUN_TIME" "$VECTOR_SIZE"
        $TIME ./bin/parallel_openmp.x "$RUN_TIME" "$VECTOR_SIZE"
        $TIME ./bin/parallel_openmp_simd.x "$RUN_TIME" "$VECTOR_SIZE"


        $TIME ./bin/serial_naive_32.x "$RUN_TIME" "$VECTOR_SIZE"
        #$TIME ./bin/serial_stl_transform_32.x "$RUN_TIME" "$VECTOR_SIZE"
        #$TIME ./bin/parallel_stl_transform_32.x "$RUN_TIME" "$VECTOR_SIZE"
        $TIME ./bin/parallel_thread_32.x "$RUN_TIME" "$VECTOR_SIZE"
        $TIME ./bin/parallel_openmp_32.x "$RUN_TIME" "$VECTOR_SIZE"
        $TIME ./bin/parallel_openmp_simd_32.x "$RUN_TIME" "$VECTOR_SIZE"

    done

} 2>&1 | tee results_cpu.log
