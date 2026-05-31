#!/bin/bash

#find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" \) ! -name "json.hpp" -exec clang-format -i {} +

clang-tidy src/serial_naive.cpp \
  -checks='clang-analyzer-*,modernize-*,performance-*' \
  -- \
  -std=c++23 \
  -Iinclude \
  -isystem include/nlohmann \
  -Wno-error
