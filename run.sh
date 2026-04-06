#!/bin/bash

docker run --gpus all -it --rm \
  -v "$PWD":/workspace \
  acpp-cuda-omp-4 \
  /bin/bash




