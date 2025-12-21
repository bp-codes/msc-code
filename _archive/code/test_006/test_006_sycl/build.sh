#!/bin/bash
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xclang -sycl-std=2020 main.cpp -o main.x
./main.x

