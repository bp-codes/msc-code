#!/bin/bash

# Pick the right SM for your GPU; sm_86 is common for RTX 30xx
export ACPP_TARGETS="cuda:sm_86"

acpp -O2 -std=c++17 example_sycl.cpp -o example_sycl

./example_sycl


