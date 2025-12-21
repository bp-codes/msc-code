#!/bin/bash

set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{

    export OMP_NUM_THREADS=4    
    /usr/bin/time -v -- ./sycl.x 5.0 1000000 add

}

