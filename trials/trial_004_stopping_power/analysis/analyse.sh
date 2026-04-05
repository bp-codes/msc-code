#!/bin/bash

source ~/venv.sh

python3 python/analyse.py --results ../results --analysis analysis

targets=(
    serial
    serial_32
    openmp
    openmp_32
    sycl
    sycl_32
    cuda
    cuda_32
)

for dir in "${targets[@]}"; do
    echo "Processing $dir"

    [ -d "../$dir/src" ] || { echo "Skipping $dir (no src/)"; continue; }

    xml="complexity_${dir}.xml"

    files=()

    for file in ../"$dir"/src/*.hpp ../"$dir"/src/*.cpp ../"$dir"/src/*.cu; do
        case "$file" in
            *json.hpp|*helper.hpp|*helper_cuda.hpp|*Error.hpp)
                continue
                ;;
        esac
        files+=("$file")
    done

    lizard "${files[@]}" -X > "files/$xml"

    python3 python/xml_to_json.py "files/$xml"
done


for dir in "${targets[@]}"; do
    echo "Processing $dir"
    ldd ../"$dir"/bin/main.x > files/libs_${dir}.txt
done




python3 python/summarise_complexity.py files



