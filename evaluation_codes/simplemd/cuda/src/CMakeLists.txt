
# App binary
add_executable(SimpleMD.x main.cpp)

target_include_directories(SimpleMD.x
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include     
    SYSTEM PRIVATE  
        ${CMAKE_SOURCE_DIR}/external/json     # NLohmann JSON
)


# AdaptiveCpp flags (compile + link)
set(ACPP_TARGETS "cuda:sm_86" CACHE STRING "AdaptiveCpp targets")

target_compile_options(SimpleMD.x PRIVATE
    -O3
    -v
    -ffast-math
    --acpp-targets=${ACPP_TARGETS}
)

target_link_options(SimpleMD.x PRIVATE
    -O3
    -v 
    -ffast-math
    --acpp-targets=${ACPP_TARGETS}
)

