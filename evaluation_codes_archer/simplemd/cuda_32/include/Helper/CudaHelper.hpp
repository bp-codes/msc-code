#ifndef CUDA_HELPER_HPP
#define CUDA_HELPER_HPP


#ifdef __CUDACC__
    #define SIMPLEMD_HOST_DEVICE __host__ __device__
#else
    #define SIMPLEMD_HOST_DEVICE
#endif

#endif
