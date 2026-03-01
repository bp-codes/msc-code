#!/bin/bash

set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{
 
    ./sycl.x 5.0 1000000
    ./cuda.x 5.0 1000000

}

