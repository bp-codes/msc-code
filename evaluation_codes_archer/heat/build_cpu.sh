#!/bin/bash

cd serial
./build.sh
cd ../

cd serial_32
./build.sh
cd ../

cd openmp
./build.sh
cd ../

cd openmp_32
./build.sh
cd ../


