#!/bin/bash

#  ./bin/sycl.x 5.0 1000000 add
#  ./bin/sycl_32.x 5.0 1000000 add
#  ./bin/cuda.x 5.0 1000000 add
#  ./bin/opencl.x 5.0 1000000 add

#apt update
#apt install time
#apt install pocl-opencl-icd ocl-icd-libopencl1 clinfo
#set -euo pipefail

set -euo pipefail
shopt -s nocasematch
RUNS=5
mkdir -p results



# Each entry: "executable arguments"
apps=(
    "./bin/parallel_cuda.x"
    "./bin/parallel_cuda_32.x"
    "./bin/parallel_sycl.x"
    "./bin/parallel_sycl_32.x"
    "./bin/parallel_opencl.x"
    "./bin/parallel_opencl_32.x"
)

operations=(
    "add"
    "multiply"
    "divide"
    "power"
    "exp"
    "log"
    "sqrt"
)

#./bin/serial.x 5.0 1000000 add

for app in "${apps[@]}"
do
    for op in "${operations[@]}"
    do
        echo "Running: $app ($op)"

        for ((i=1; i<=RUNS; i++))
        do
            echo "  Run $i"

            if [[ "$app" == *opencl* || "$app" == *sycl* ]]; then

                echo "    CPU"
                CMD="$app 5.0 1000000 $op CPU"
                echo $CMD
                eval $CMD

                echo "    GPU"
                CMD="$app 5.0 1000000 $op GPU"
                echo $CMD
                eval $CMD

            else

                CMD="$app 5.0 1000000 $op"
                echo $CMD
                eval $CMD

            fi
        done

    done
done


