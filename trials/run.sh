#!/bin/bash

docker run --gpus all -it --rm \
  -v "$PWD/../":/workspace \
  acpp-cuda-omp \
  /bin/bash




