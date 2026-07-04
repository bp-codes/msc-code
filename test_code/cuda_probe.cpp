#include <dlfcn.h>

#include <iostream>

using CUresult = int;
using cuInit_t = CUresult (*)(unsigned int);
using cuDeviceGetCount_t = CUresult (*)(int*);

int main()
{
    void* libcuda = dlopen("libcuda.so.1", RTLD_NOW);

    if (!libcuda) {
        std::cerr << "dlopen libcuda.so.1 failed: " << dlerror() << '\n';
        return 1;
    }

    auto cuInit =
        reinterpret_cast<cuInit_t>(dlsym(libcuda, "cuInit"));

    auto cuDeviceGetCount =
        reinterpret_cast<cuDeviceGetCount_t>(dlsym(libcuda, "cuDeviceGetCount"));

    if (!cuInit || !cuDeviceGetCount) {
        std::cerr << "Failed to load CUDA driver symbols\n";
        return 1;
    }

    const CUresult init_result = cuInit(0);

    int count = -1;
    const CUresult count_result = cuDeviceGetCount(&count);

    std::cout << "cuInit result: " << init_result << '\n';
    std::cout << "cuDeviceGetCount result: " << count_result << '\n';
    std::cout << "device count: " << count << '\n';

    dlclose(libcuda);

    return 0;
}
