# cmake/acpp.toolchain.cmake
set(CMAKE_CXX_COMPILER acpp CACHE FILEPATH "")

# You can also set C if you build C code:
# set(CMAKE_C_COMPILER acpp CACHE FILEPATH "")

# Default build type if not provided
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# AdaptiveCpp target
set(ACPP_TARGETS "cuda:sm_86" CACHE STRING "AdaptiveCpp targets")

# Match your CLI flags
add_compile_options(
  -O3
  -v
  -fopenmp
  --acpp-targets=${ACPP_TARGETS}
)

# The link step also needs these for acpp
add_link_options(
  -v
  -fopenmp
  --acpp-targets=${ACPP_TARGETS}
)
 
