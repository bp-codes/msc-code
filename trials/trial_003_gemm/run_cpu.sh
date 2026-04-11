#!/bin/bash
export OMP_NUM_THREADS=6

mkdir -p results

RUNS=5
TIMER=3.0

executables=(
    ./bin/serial_naive.x
    ./bin/serial_optimized.x
    ./bin/parallel_openmp.x
    ./bin/parallel_blas.x
)

sizes=(
    "128 128 128"
    "1000 1200 800"
    "1000 1000 1000"
    "1024 1024 1024"
    "4096 4096 4096"
)

for size in "${sizes[@]}"; do
    #./bin/precise.x $TIMER $size
    #echo
done

for ((i=1; i<=RUNS; i++))
do
    for size in "${sizes[@]}"; do
        for exe in "${executables[@]}"; do
            $exe $TIMER $size
        done
        echo
    done
done





