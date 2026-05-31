#!/bin/bash

source ~/venv.sh
python3 python/performance.py  --results ../results

python3 python/precision.py ../serial/output/out.xyz ../serial_32/output/out.xyz --outdir=precision/serial_32 --plot_name serial_32
python3 python/precision.py ../serial/output/out.xyz ../openmp/output/out.xyz --outdir=precision/openmp --plot_name openmp
python3 python/precision.py ../serial/output/out.xyz ../openmp_32/output/out.xyz --outdir=precision/openmp_32 --plot_name openmp_32
python3 python/precision.py ../serial/output/out.xyz ../sycl/output_cpu/out.xyz --outdir=precision/sycl --plot_name sycl_cpu
python3 python/precision.py ../serial/output/out.xyz ../sycl/output_gpu/out.xyz --outdir=precision/sycl --plot_name sycl_gpu
python3 python/precision.py ../serial/output/out.xyz ../sycl_32/output_cpu/out.xyz --outdir=precision/sycl_32 --plot_name sycl_32_cpu
python3 python/precision.py ../serial/output/out.xyz ../sycl_32/output_gpu/out.xyz --outdir=precision/sycl_32 --plot_name sycl_32_gpu




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



