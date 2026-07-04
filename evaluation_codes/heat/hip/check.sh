#!/bin/bash
#set -Eeuo pipefail

# Run cppcheck
trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{

    find src \( -name "*.cpp" -o -name "*.hpp" -o -name '*.cu' \) -print0 | \
    xargs -0 -n1 cppcheck --language=c++ -I include \
        --enable=warning,style,performance \
        --quiet \
        --suppress=syntaxError:include/nlohmann/json.hpp \
        --suppress=missingIncludeSystem \
        --suppress=toomanyconfigs

    find include \( -name "*.cpp" -o -name "*.hpp" -o -name '*.cu' \) -print0 | \
    xargs -0 -n1 cppcheck --language=c++ -I include \
        --enable=warning,style,performance \
        --quiet \
        --suppress=syntaxError:include/nlohmann/json.hpp \
        --suppress=missingIncludeSystem \
        --suppress=toomanyconfigs        

} 2>&1 | tee cppcheck.log

# Run cpplint
trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{

    find src -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' \) \
        ! -name 'json.hpp' \
        -exec python3 ../../../external/cpplint/cpplint.py \
      --filter=-build/c++17,+build/include_order,-whitespace/indent \
          --linelength=120 \
          {} +

    find include -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' \) \
        ! -name 'json.hpp' \
        -exec python3 ../../../external/cpplint/cpplint.py \
      --filter=-build/c++17,+build/include_order,-whitespace/indent \
          --linelength=120 \
          {} +

} 2>&1 | tee cpplint.log
