#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

    export OMP_NUM_THREADS=4


    /usr/bin/time -v -- ./serial.x 5.0 1000000
    /usr/bin/time -v -- ./openmp.x 5.0 1000000

} 2>&1 | tee results_cpu.log


