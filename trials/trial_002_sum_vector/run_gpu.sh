#!/bin/bash
export ACPP_DISABLE_OCL=1
./sycl.x 10.0 10000000 256 gpu
./sycl_reduction.x 10.0 10000000 256 gpu
./cuda.x 10.0 10000000




