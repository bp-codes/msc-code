#!/bin/bash
export OMP_NUM_THREADS=6
./bin/main.x input_cpu.json
./bin/main.x input_gpu.json


