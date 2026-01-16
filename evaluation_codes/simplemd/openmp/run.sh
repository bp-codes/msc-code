#!/bin/bash
set -Eeuo pipefail

trap 'echo "❌ Error on line $LINENO (exit code $?)" >&2' ERR

{


    echo "Test Run"
    ./SimpleMD.x input.json

} 2>&1 | tee build/build.log
