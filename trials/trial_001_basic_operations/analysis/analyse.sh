#!/bin/bash

source ~/venv.sh

python3 python/analyse.py --results ../results --analysis analysis --trial "Basic Operations" --selected_operation "add"
python3 python/analyse.py --results ../results --analysis analysis --trial "Basic Operations" --selected_operation "divide"
python3 python/analyse.py --results ../results --analysis analysis --trial "Basic Operations" --selected_operation "exp"
python3 python/analyse.py --results ../results --analysis analysis --trial "Basic Operations" --selected_operation "log"
python3 python/analyse.py --results ../results --analysis analysis --trial "Basic Operations" --selected_operation "multiply"
python3 python/analyse.py --results ../results --analysis analysis --trial "Basic Operations" --selected_operation "power"
python3 python/analyse.py --results ../results --analysis analysis --trial "Basic Operations" --selected_operation "sqrt"
