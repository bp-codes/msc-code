#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

    export NUM_THREADS=6


    /usr/bin/time -v -- ./bin/serial.x 5.0 1000000
    /usr/bin/time -v -- ./bin/parallel_openmp.x 5.0 1000000

} 2>&1 | tee results_cpu.log


