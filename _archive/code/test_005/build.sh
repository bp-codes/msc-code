#!/bin/bash
icpx -fsycl -O2 -std=c++17 dev.cpp -o dev.x
./dev.x



#g++ -std=c++17 -O2 -fopenmp -ltbb dev.cpp -o dev.x
