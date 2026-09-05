cmake_minimum_required(VERSION 3.15)

project(SimpleMD
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)

if(BUILD_TESTING)
    #set(INSTALL_GTEST OFF CACHE BOOL "Disable gtest install" FORCE)
    #set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    #add_subdirectory(external/googletest)
    #add_subdirectory(src_tests)
endif()
