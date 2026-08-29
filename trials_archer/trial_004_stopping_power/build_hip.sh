#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{
    mkdir -p bin

    # HIP
    hipcc -std=c++20 -O3 \
        --gpu-architecture=sm_86 \
         src/parallel_hip.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_hip.x

    # HIP 32
    hipcc -std=c++20 -O3 \
        --gpu-architecture=sm_86 \
         src/parallel_hip_32.cpp \
        -Iinclude \
        -isystem include/nlohmann \
        -o bin/parallel_hip_32.x

} 2>&1 | tee build.log
