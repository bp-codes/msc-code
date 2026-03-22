#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR

{
    mkdir -p build
    cd build

    cmake ..

    cmake --build . -- -j

    cd ../
    ctest --test-dir build -V

    echo "Copy"
    mkdir -p bin
    cp build/src/SimpleMD.x bin/SimpleMD.x


} 2>&1 | tee build/build.log
