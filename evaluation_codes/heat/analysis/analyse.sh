#!/bin/bash

source ~/venv.sh
python3 python/performance.py  --results ../results

python3 python/precision.py --framework serial --csv_dir output --plot_name serial
python3 python/precision.py --framework serial_32 --csv_dir output --plot_name serial./analyse.sh

python3 python/precision.py --framework openmp --csv_dir output --plot_name openmp
python3 python/precision.py --framework openmp_32 --csv_dir output --plot_name openmp_32

python3 python/precision.py --framework cuda --csv_dir output --plot_name cuda
python3 python/precision.py --framework cuda_32 --csv_dir output --plot_name cuda_32

python3 python/precision.py --framework sycl --csv_dir output_cpu --plot_name sycl_cpu
python3 python/precision.py --framework sycl --csv_dir output_gpu --plot_name sycl_gpu

python3 python/precision.py --framework sycl_32 --csv_dir output_cpu --plot_name sycl_32_cpu
python3 python/precision.py --framework sycl_32 --csv_dir output_gpu --plot_name sycl_32_gpu



exit 0

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



