#!/bin/bash

g++ -O3 -std=c++17 -fopenmp data_race_sum.cpp -o data_race_sum.x
g++ -O3 -std=c++17 -fopenmp data_race_sum_critical.cpp -o data_race_sum_critical.x
g++ -O3 -std=c++17 -fopenmp data_race_sum_atomic.cpp -o data_race_sum_atomic.x
g++ -O3 -std=c++17 -fopenmp data_race_sum_reduction.cpp -o data_race_sum_reduction.x
./data_race_sum.x
./data_race_sum_critical.x
./data_race_sum_atomic.x
./data_race_sum_reduction.x
