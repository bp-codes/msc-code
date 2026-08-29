#!/bin/bash
#docker run -it --gpus all -v $(pwd):/workspace sycl-base:ubuntu22 bash 
#docker run --rm -it --gpus all -v $(pwd):/workspace gpu-stack:latest bash -lc "
#  nvidia-smi && \
#  nvcc --version && \
#  g++ -fopenmp -dM -E - < /dev/null | grep _OPENMP || true && \
#  clinfo -l && \
#  icpx -fsycl --version
#"

docker run -it --gpus all -v $(pwd):/workspace gpu-stack:jammy bash

