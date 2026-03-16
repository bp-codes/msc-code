#!/bin/bash

# Precise

g++ -O3 -ffast-math -march=native -std=c++23 -fext-numeric-literals src/precise.cpp  -lquadmath -o bin/precise.x

# 64 bit

g++ -O3 -ffast-math -march=native -std=c++23 src/serial_naive.cpp -o bin/serial_naive.x
g++ -O3 -ffast-math -march=native -std=c++23 -mavx2 -mfma src/serial_simd.cpp -o bin/serial_simd.x
g++ -O3 -ffast-math -march=native -std=c++23 src/serial_stl_accumulate.cpp -o bin/serial_stl_accumulate.x
g++ -O3 -ffast-math -march=native -std=c++23 src/serial_stl_reduce.cpp -o bin/serial_stl_reduce.x
g++ -O3 -ffast-math -march=native -std=c++23 src/serial_stl_transform_reduce.cpp -o bin/serial_stl_transform_reduce.x
  
g++ -O3 -ffast-math -march=native -std=c++23 src/parallel_stl_reduce.cpp -ltbb -o bin/parallel_stl_reduce.x
g++ -O3 -ffast-math -march=native -std=c++23 src/parallel_stl_transform_reduce.cpp -ltbb -o bin/parallel_stl_transform_reduce.x
g++ -O3 -ffast-math -march=native -std=c++23 src/parallel_thread.cpp -ltbb -o bin/parallel_thread.x
g++ -O3 -ffast-math -march=native -std=c++23 -mavx2 -mfma src/parallel_thread_simd.cpp -ltbb -o bin/parallel_thread_simd.x
g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp.cpp -o bin/parallel_openmp.x
g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp_simd.cpp -o bin/parallel_openmp_simd.x
g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp_tree.cpp -o bin/parallel_openmp_tree.x
g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp_reduction.cpp -o bin/parallel_openmp_reduction.x
g++ -O3 -ffast-math -march=native -std=c++23 -fopenmp src/parallel_openmp_reduction_simd.cpp -o bin/parallel_openmp_reduction_simd.x

# 32 bit

g++ -O3 -ffast-math -march=native -std=c++23 src/serial_naive_32.cpp -o bin/serial_naive_32.x
g++ -O3 -ffast-math -march=native -std=c++23 -mavx2 -mfma src/serial_simd_32.cpp -o bin/serial_simd_32.x











