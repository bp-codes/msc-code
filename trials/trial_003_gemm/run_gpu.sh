#!/bin/bash
export OMP_NUM_THREADS=6
export SYCL_DEVICE_FILTER=cuda

mkdir -p results

RUNS=2
TIMER=3.0

executables=(
    ./bin/parallel_cuda.x
    ./bin/parallel_cuda_cublas.x
    ./bin/parallel_sycl.x
    ./bin/parallel_cuda_32.x
    ./bin/parallel_cuda_cublas_32.x
    ./bin/parallel_sycl_32.x
    ./bin/parallel_opencl.x
)

sizes=(
    "128 128 128"
    "1000 1200 800"
    "1000 1000 1000"
    "1024 1024 1024"
    "4096 4096 4096"
)

for size in "${sizes[@]}"; do
    CMD="./bin/precise.x $TIMER $size"
    #eval $CMD
    echo $CMD
done

for ((i=1; i<=RUNS; i++))
do
    for size in "${sizes[@]}"; do
        for exe in "${executables[@]}"; do
            CMD="$exe $TIMER $size"
            echo $CMD
            eval $CMD
        done
        echo
    done
    echo
done




