#!/bin/bash

g++ -O3 -ffast-math -march=native -std=c++17 serial_naive.cpp -o serial_naive.x
g++ -O3 -ffast-math -march=native -std=c++17 serial_stl_reduce.cpp -o serial_stl_reduce.x
g++ -O3 -ffast-math -march=native -std=c++17 serial_stl_accumulate.cpp -o serial_stl_accumulate.x
g++ -O3 -ffast-math -march=native -std=c++17 -mavx2 -mfma serial_simd.cpp -o serial_simd.x
  
g++ -O3 -ffast-math -march=native -std=c++17 -fopenmp openmp.cpp -o openmp.x
g++ -O3 -ffast-math -march=native -std=c++17 -fopenmp openmp_simd.cpp -o openmp_simd.x
g++ -O3 -ffast-math -march=native -std=c++17 -fopenmp openmp_tree.cpp -o openmp_tree.x




