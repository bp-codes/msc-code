# Header-only library
add_library(simplemd INTERFACE)
target_include_directories(simplemd
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ${CMAKE_SOURCE_DIR}/external/json     # NLohmann JSON
)

add_library(SimpleMD::simplemd ALIAS simplemd)

# App binary
add_executable(SimpleMD.x main.cpp)

# C++ standard
target_compile_features(SimpleMD.x PRIVATE cxx_std_17)

# AdaptiveCpp flags (compile + link)
set(ACPP_TARGETS "cuda:sm_86" CACHE STRING "AdaptiveCpp targets")

target_compile_options(SimpleMD.x PRIVATE
    -O3
    -v
    -fopenmp
    --acpp-targets=${ACPP_TARGETS}
)

target_link_options(SimpleMD.x PRIVATE
    -v
    -fopenmp
    --acpp-targets=${ACPP_TARGETS}
)

target_link_libraries(SimpleMD.x PRIVATE SimpleMD::simplemd)
