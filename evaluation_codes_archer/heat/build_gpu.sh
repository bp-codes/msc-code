#!/bin/bash

cd cuda
./build.sh
cd ../

cd cuda_32
./build.sh
cd ../

cd sycl
./build.sh
cd ../

cd sycl_32
./build.sh
cd ../


