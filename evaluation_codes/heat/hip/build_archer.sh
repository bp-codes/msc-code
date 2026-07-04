#!/bin/bash
mkdir -p bin

module restore
module load PrgEnv-amd
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan

hipcc -x hip -O3 -std=c++20 \
      -D__HIP_ROCclr__ \
      --rocm-path="${ROCM_PATH}" \
      -D__HIP_PLATFORM_AMD__ \
      --offload-arch=gfx90a \
    -Iinclude \
    -isystem include/nlohmann -Wno-nan-infinity-disabled \
      src/main.cpp -o bin/main.x
