#!/bin/bash

docker run --gpus all -it --rm \
  -v "$PWD":/workspace \
  hip-nvidia \
  /bin/bash
