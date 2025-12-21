#!/bin/bash
nvcc -Xcompiler -lcublas -O2 main.cu -o main.x


cd test_001_nvcc && ./build.sh && cd ../
cd test_001_nvcc && ./build.sh && cd ../

