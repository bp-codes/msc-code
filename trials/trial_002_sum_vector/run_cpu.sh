#!/bin/bash
export NUM_THREADS=6

RUNS=5

./bin/precise.x 10.0 1000000

for ((i=1; i<=RUNS; i++))
do

    # Serial
    ./bin/serial_naive.x 10.0 1000000
    ./bin/serial_simd.x 10.0 1000000
    ./bin/serial_stl_accumulate.x 10.0 1000000
    ./bin/serial_stl_reduce.x 10.0 1000000
    ./bin/serial_stl_transform_reduce.x 10.0 1000000

    # Parallel CPU
    ./bin/parallel_stl_reduce.x 10.0 1000000
    ./bin/parallel_stl_transform_reduce.x 10.0 1000000
    ./bin/parallel_thread.x 10.0 1000000
    ./bin/parallel_thread_simd.x 10.0 1000000
    ./bin/parallel_openmp.x 10.0 1000000
    ./bin/parallel_openmp_simd.x 10.0 1000000
    ./bin/parallel_openmp_tree.x 10.0 1000000
    ./bin/parallel_openmp_reduction.x 10.0 1000000
    ./bin/parallel_openmp_reduction_simd.x 10.0 1000000

    # 32 bit

    # Serial
    ./bin/serial_naive_32.x 10.0 1000000
    ./bin/serial_simd_32.x 10.0 1000000

done

#./openmp.x 10.0 1000000
#./openmp_tree.x 10.0 1000000
