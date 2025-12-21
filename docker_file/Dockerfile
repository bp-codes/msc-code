FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git ninja-build \
    python3 python3-pip python3-setuptools python3-wheel \
    pkg-config wget ca-certificates gnupg \
    zlib1g-dev libzstd-dev libtinfo-dev libxml2-dev libedit-dev libncurses5-dev \
 && rm -rf /var/lib/apt/lists/*


RUN set -eux; \
    wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor > /usr/share/keyrings/llvm.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
      > /etc/apt/sources.list.d/llvm.list; \
    apt-get update; \
    apt-get install -y \
      clang-18 llvm-18 llvm-18-dev llvm-18-tools \
      libclang-18-dev lld-18 \
      libomp-18-dev libomp5-18; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --recurse-submodules https://github.com/AdaptiveCpp/AdaptiveCpp.git

RUN set -eux; \
    CLANG_BIN="$(command -v clang++-18)"; \
    LLVM_DIR="$(llvm-config-18 --cmakedir)"; \
    CLANG_RESOURCE_DIR="$("$CLANG_BIN" -print-resource-dir)"; \
    test -f "$CLANG_RESOURCE_DIR/include/__clang_cuda_runtime_wrapper.h"; \
    cmake -S /opt/AdaptiveCpp -B /opt/AdaptiveCpp/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/acpp \
      -DWITH_CUDA_BACKEND=ON \
      -DWITH_ROCM_BACKEND=OFF \
      -DWITH_CPU_BACKEND=ON \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DLLVM_DIR="$LLVM_DIR" \
      -DCLANG_EXECUTABLE_PATH="$CLANG_BIN" \
      -DCLANG_INCLUDE_PATH="$CLANG_RESOURCE_DIR/include" \
      -DACPP_LLD_PATH="$(command -v ld.lld-18)" \
    && cmake --build /opt/AdaptiveCpp/build --target install -j"$(nproc)"

ENV PATH="/opt/acpp/bin:${PATH}"
CMD ["/bin/bash"]
