#!/bin/bash
set -Eeuo pipefail
export OMP_NUM_THREADS=6

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{
    echo "Test Run"
    ./bin/SimpleMD.x input.json

} 2>&1 | tee build/build.log