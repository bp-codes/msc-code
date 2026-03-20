#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{
    ./bin/precise.x 5.0 1000000

    RUNS=5
    export NUM_THREADS=6

    for ((i=1; i<=RUNS; i++))
    do
        echo "  Run $i"

        /usr/bin/time -v -- ./bin/serial.x 5.0 1000000
        /usr/bin/time -v -- ./bin/parallel_transform.x 5.0 1000000
        /usr/bin/time -v -- ./bin/parallel_thread.x 5.0 1000000
        /usr/bin/time -v -- ./bin/parallel_openmp.x 5.0 1000000
        /usr/bin/time -v -- ./bin/parallel_openmp_simd.x 5.0 1000000


        /usr/bin/time -v -- ./bin/serial_32.x 5.0 1000000
        /usr/bin/time -v -- ./bin/parallel_transform_32.x 5.0 1000000
        /usr/bin/time -v -- ./bin/parallel_thread_32.x 5.0 1000000
        /usr/bin/time -v -- ./bin/parallel_openmp_32.x 5.0 1000000
        /usr/bin/time -v -- ./bin/parallel_openmp_simd_32.x 5.0 1000000

    done

} 2>&1 | tee results_cpu.log


