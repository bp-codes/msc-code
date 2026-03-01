#!/bin/bash

g++ -O3 -std=c++17 -fopenmp data_race.cpp -o data_race.x
./data_race.x
