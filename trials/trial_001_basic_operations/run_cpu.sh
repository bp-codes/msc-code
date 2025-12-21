#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

    export OMP_NUM_THREADS=4


    /usr/bin/time -v -- ./serial.x 5.0 1000000 add
    /usr/bin/time -v -- ./serial.x 5.0 1000000 add
    /usr/bin/time -v -- ./serial.x 5.0 1000000 add
    /usr/bin/time -v -- ./serial_simd.x 5.0 1000000 add
    /usr/bin/time -v -- ./serial_simd.x 5.0 1000000 add
    /usr/bin/time -v -- ./serial_simd.x 5.0 1000000 add
    /usr/bin/time -v -- ./serial_stl.x 5.0 1000000 add
    /usr/bin/time -v -- ./serial_stl.x 5.0 1000000 add
    /usr/bin/time -v -- ./serial_stl.x 5.0 1000000 add
    /usr/bin/time -v -- ./openmp.x 5.0 1000000 add
    /usr/bin/time -v -- ./openmp.x 5.0 1000000 add
    /usr/bin/time -v -- ./openmp.x 5.0 1000000 add
    /usr/bin/time -v -- ./openmp_simd.x 5.0 1000000 add
    /usr/bin/time -v -- ./openmp_simd.x 5.0 1000000 add
    /usr/bin/time -v -- ./openmp_simd.x 5.0 1000000 add

} 2>&1 | tee results.log


