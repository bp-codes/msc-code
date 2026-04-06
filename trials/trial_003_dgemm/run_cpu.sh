#!/bin/bash
export OMP_NUM_THREADS=6

mkdir -p results

./bin/serial_naive.x 60.0 128 128 128
./bin/serial_optimized.x 60.0 128 128 128
./bin/parallel_openmp.x 60.0 128 128 128
./bin/parallel_blas.x 60.0 128 128 128



./bin/serial_naive.x 60.0 1000 1200 800
./bin/serial_optimized.x 60.0 1000 1200 800
./bin/parallel_openmp.x 60.0 1000 1200 800
./bin/parallel_blas.x 60.0 1000 1200 800



./bin/serial_naive.x 60.0 1000 1000 1000
./bin/serial_optimized.x 60.0 1000 1000 1000
./bin/parallel_openmp.x 60.0 1000 1000 1000
./bin/parallel_blas.x 60.0 1000 1000 1000



./bin/serial_naive.x 60.0 4096 4096 4096
./bin/serial_optimized.x 60.0 4096 4096 4096
./bin/parallel_openmp.x 60.0 4096 4096 4096
./bin/parallel_blas.x 60.0 4096 4096 4096






