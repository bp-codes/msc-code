

# Compile for NVIDIA offload with Clang/LLVM
export PATH=/usr/local/cuda/bin:$PATH
export OMP_TARGET_OFFLOAD=MANDATORY

clang++ -O2 -std=c++17 -fopenmp \
  -fopenmp-targets=nvptx64-nvidia-cuda \
  --offload-arch=sm_86 \
  --cuda-path=/usr/local/cuda \
  saxpy_omp.cpp -o saxpy_omp

# Helpful runtime settings
export OMP_TARGET_OFFLOAD=MANDATORY      # fail if GPU isn’t used
# export LIBOMPTARGET_INFO=4             # verbose offload logs (optional)

./saxpy_omp
