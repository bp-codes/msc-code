#!/bin/bash

find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" \) ! -name "json.hpp" -exec clang-format -i {} +
