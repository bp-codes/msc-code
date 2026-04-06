#!/bin/bash

find src \( -name "*.cpp" -o -name "*.hpp" \) -print0 | \
xargs -0 -n1 cppcheck -I src \
  --enable=warning,style,performance \
  --quiet \
  --suppress=syntaxError:src/json.hpp \
  --suppress=missingIncludeSystem \
  --suppress=toomanyconfigs

