cmake_minimum_required(VERSION 3.15)
project(SimpleMD VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Don't require OpenMP discovery with acpp; create a fallback target.
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND)
    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
    target_compile_options(OpenMP::OpenMP_CXX INTERFACE -fopenmp)
    target_link_options(OpenMP::OpenMP_CXX INTERFACE -fopenmp -O3 -ffast-math)
endif()

add_subdirectory(src)

if(BUILD_TESTING)
    set(INSTALL_GTEST OFF CACHE BOOL "Disable gtest install" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(external/googletest)
    add_subdirectory(src_tests)
endif()
