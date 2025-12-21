#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=cuda


./sycl.x 10.0 1000 1000
./cuda.x 10.0 1000 1000

