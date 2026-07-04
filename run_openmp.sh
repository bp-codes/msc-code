#!/bin/bash

docker run --gpus all -it --rm \
  -v "$PWD":/workspace \
  docker-openmp-gpu \
  /bin/bash




