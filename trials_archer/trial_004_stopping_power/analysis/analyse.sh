#!/bin/bash

exec > >(tee analyse.log) 2>&1

source ~/venv.sh

python3 python/analyse.py --results ../results --analysis analysis --trial "Bethe-Bloch Stopping Power" --selected_operation "Bethe-Bloch Stopping Power"
