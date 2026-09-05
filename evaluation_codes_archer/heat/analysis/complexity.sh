#!/bin/bash

source ~/venv.sh

#mkdir -p complexity

#for file in ../src/*.hpp ../src/*.cpp ../src/*.cu; do
#    base=$(basename "$file")
#    xml="complexity/complexity_${base}.xml"
#    lizard "$file" -X --language=cpp > "$xml"
#done

#python3 python/complexity.py
python3 python/line_count.py
python3 python/word_count.py
python3 python/function_count.py
python3 python/character_count.py
