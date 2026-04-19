#!/bin/bash

set -euo pipefail
export NUM_THREADS=6
mkdir -p results

RUNS=3
RUN_TIME=10.0
VECTOR_SIZE=1000000

./bin/precise.x "$RUN_TIME" "$VECTOR_SIZE"

for ((i=1; i<=RUNS; i++))
do

    # 64 bit

    # Serial
    ./bin/serial_naive.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/serial_simd.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/serial_stl_accumulate.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/serial_stl_reduce.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/serial_stl_transform_reduce.x "$RUN_TIME" "$VECTOR_SIZE"

    sleep 10

    # Parallel CPU
    ./bin/parallel_stl_reduce.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_stl_transform_reduce.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_thread.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_thread_simd.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_openmp.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_openmp_simd.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_openmp_tree.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_openmp_reduction.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_openmp_reduction_simd.x "$RUN_TIME" "$VECTOR_SIZE"

    sleep 10

    # 32 bit

    # Serial
    ./bin/serial_naive_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/serial_simd_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/serial_stl_accumulate_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/serial_stl_reduce_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/serial_stl_transform_reduce_32.x "$RUN_TIME" "$VECTOR_SIZE"

    sleep 10

    # Parallel CPU
    ./bin/parallel_stl_reduce_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_stl_transform_reduce_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_thread_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_thread_simd_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_openmp_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_openmp_simd_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_openmp_tree_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_openmp_reduction_32.x "$RUN_TIME" "$VECTOR_SIZE"
    ./bin/parallel_openmp_reduction_simd_32.x "$RUN_TIME" "$VECTOR_SIZE"

    sleep 10
    
done

#./openmp.x 10.0 1000000
#./openmp_tree.x 10.0 1000000
