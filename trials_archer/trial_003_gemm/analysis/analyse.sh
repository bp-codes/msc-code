#!/bin/bash

source ~/venv.sh

python3 python/analyse.py --results ../results --analysis analysis --trial "GEMM" --selected_operation "128x128_by_128x128"
python3 python/analyse.py --results ../results --analysis analysis --trial "GEMM" --selected_operation "1000x800_by_800x1200"
python3 python/analyse.py --results ../results --analysis analysis --trial "GEMM" --selected_operation "1024x1024_by_1024x1024"
python3 python/analyse.py --results ../results --analysis analysis --trial "GEMM" --selected_operation "4096x4096_by_4096x4096"
