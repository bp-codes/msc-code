#!/bin/bash
icpx -fsycl -O2 -std=c++17 main.cpp -o main.x
./main.x

