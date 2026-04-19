#!/bin/bash
set -Eeuo pipefail
export NUM_THREADS=6
mkdir -p results

RUNS=3
RUN_TIME=10.0
VECTOR_SIZE=1000000

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{
    ./bin/precise.x "$RUN_TIME" "$VECTOR_SIZE"

    export NUM_THREADS=6

    for ((i=1; i<=RUNS; i++))
    do
        echo "  Run $i"

        /usr/bin/time -v -- ./bin/serial.x "$RUN_TIME" "$VECTOR_SIZE"
        /usr/bin/time -v -- ./bin/serial_stl.x "$RUN_TIME" "$VECTOR_SIZE"
        /usr/bin/time -v -- ./bin/parallel_transform.x "$RUN_TIME" "$VECTOR_SIZE"
        /usr/bin/time -v -- ./bin/parallel_stl.x "$RUN_TIME" "$VECTOR_SIZE"
        /usr/bin/time -v -- ./bin/parallel_thread.x "$RUN_TIME" "$VECTOR_SIZE"
        /usr/bin/time -v -- ./bin/parallel_openmp.x "$RUN_TIME" "$VECTOR_SIZE"
        /usr/bin/time -v -- ./bin/parallel_openmp_simd.x "$RUN_TIME" "$VECTOR_SIZE"


        /usr/bin/time -v -- ./bin/serial_32.x "$RUN_TIME" "$VECTOR_SIZE"
        /usr/bin/time -v -- ./bin/parallel_transform_32.x "$RUN_TIME" "$VECTOR_SIZE"
        /usr/bin/time -v -- ./bin/parallel_thread_32.x "$RUN_TIME" "$VECTOR_SIZE"
        /usr/bin/time -v -- ./bin/parallel_openmp_32.x "$RUN_TIME" "$VECTOR_SIZE"
        /usr/bin/time -v -- ./bin/parallel_openmp_simd_32.x "$RUN_TIME" "$VECTOR_SIZE"

    done

} 2>&1 | tee results_cpu.log


