#!/bin/bash
export OMP_NUM_THREADS=4
/usr/bin/time -v -- ./serial_naive.x 10.0 10000000
/usr/bin/time -v -- ./serial_stl_reduce.x 10.0 10000000
/usr/bin/time -v -- ./serial_stl_accumulate.x 10.0 10000000
/usr/bin/time -v -- ./serial_simd.x 10.0 10000000

/usr/bin/time -v -- ./openmp.x 10.0 10000000
/usr/bin/time -v -- ./openmp_simd.x 10.0 10000000
/usr/bin/time -v -- ./openmp_tree.x 10.0 10000000

#./openmp.x 10.0 10000000
#./openmp_tree.x 10.0 10000000
