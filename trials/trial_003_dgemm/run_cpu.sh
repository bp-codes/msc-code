#!/bin/bash
export OMP_NUM_THREADS=6
./serial.x 10.0 1000 1000
./openmp.x 10.0 1000 1000