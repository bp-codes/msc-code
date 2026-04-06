#!/bin/bash
set -Eeuo pipefail

trap 'echo "Error on line $LINENO (exit code $?)" >&2' ERR
{

  find src \( -name "*.cpp" -o -name "*.hpp" \) -print0 | \
  xargs -0 -n1 cppcheck -I src \
    --enable=warning,style,performance \
    --quiet \
    --suppress=syntaxError:src/json.hpp \
    --suppress=missingIncludeSystem \
    --suppress=toomanyconfigs

} 2>&1 | tee check.log

