#!/bin/bash

cd cuda
./run.sh
cd ../

cd cuda_32
./run.sh
cd ../

cd sycl
./run.sh
cd ../

cd sycl_32
./run.sh
cd ../

cd opencl
./run.sh
cd ../

cd opencl_32
./run.sh
cd ../


