#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{
    module restore
    module load PrgEnv-amd
    module load rocm
    module load craype-accel-amd-gfx90a
    module load craype-x86-milan

    mkdir -p bin

    # HIP
    hipcc -x hip -O3 -std=c++20 \
        -D__HIP_ROCclr__ \
        --rocm-path="${ROCM_PATH}" \
        -D__HIP_PLATFORM_AMD__ \
        --offload-arch=gfx90a \
        -Iinclude \
        -isystem include/nlohmann -Wno-nan-infinity-disabled \
        src/parallel_hip.cpp -o bin/parallel_hip.x


    # HIP 32
    hipcc -x hip -O3 -std=c++20 \
        -D__HIP_ROCclr__ \
        --rocm-path="${ROCM_PATH}" \
        -D__HIP_PLATFORM_AMD__ \
        --offload-arch=gfx90a \
        -Iinclude \
        -isystem include/nlohmann -Wno-nan-infinity-disabled \
        src/parallel_hip_32.cpp -o bin/parallel_hip_32.x

} 2>&1 | tee build_archer_hip.log
