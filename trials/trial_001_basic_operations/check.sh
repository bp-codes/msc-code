#!/bin/bash
set -Eeuo pipefail

# Run cppcheck
trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{

    find src \( -name "*.cpp" -o -name "*.hpp" -o -name '*.cu' \) -print0 | \
    xargs -0 -n1 cppcheck --language=c++ -I src \
        --enable=warning,style,performance \
        --quiet \
        --suppress=syntaxError:src/json.hpp \
        --suppress=missingIncludeSystem \
        --suppress=toomanyconfigs

} 2>&1 | tee cppcheck.log

# Run cpplint
trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{

    find src -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' \) \
        ! -name 'json.hpp' \
        -exec python3 ../../external/cpplint/cpplint.py \
          --filter=-build/+build/include_order,-whitespace/indent \
          --linelength=120 \
          {} +

} 2>&1 | tee cpplint.log
