#!/bin/bash
export OMP_NUM_THREADS=6
./bin/heat2d.x input_cpu.json
./bin/heat2d.x input_gpu.json


