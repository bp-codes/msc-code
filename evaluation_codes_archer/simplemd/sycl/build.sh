#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

status=0

{
    # Configure
    cmake -S . -B build \
        -DCMAKE_TOOLCHAIN_FILE=acpp.toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release

    # Build
    cmake --build build -- -j1

    # Run tests
    ctest --test-dir build -V

    # Copy executable
    echo "Copy"
    mkdir -p bin
    cp build/src/SimpleMD.x bin/SimpleMD.x

} 2>&1 | tee build.log || status=$?

if grep -q -E "error:" build.log; then
    echo
    echo "==== ERRORS ===="
    grep -n -E "error:" build.log
fi

exit "$status"