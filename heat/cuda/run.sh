#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda
OMP_NUM_THREADS=6
./main.x input.json


