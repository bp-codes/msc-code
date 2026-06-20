#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda
export OMP_NUM_THREADS=6
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

    echo "Test Run"
    ./bin/SimpleMD.x input.json

} 2>&1 | tee build/build.log
