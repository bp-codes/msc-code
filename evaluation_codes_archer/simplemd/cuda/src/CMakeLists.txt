
# App binary
add_executable(SimpleMD.x
    main.cpp
    ConfigurationEngine.cu
    VerletEngine.cu
)

target_include_directories(SimpleMD.x
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include     
    SYSTEM PRIVATE  
        ${CMAKE_SOURCE_DIR}/external/json     # NLohmann JSON
)

target_link_libraries(SimpleMD.x
    PRIVATE
        CUDA::cudart
)

target_compile_options(SimpleMD.x PRIVATE

    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
    >

    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -Xcompiler=-fno-fast-math
        -Xcompiler=-fno-unsafe-math-optimizations
        -Xcompiler=-ffp-contract=off
    >
)

set_target_properties(SimpleMD.x PROPERTIES
    CUDA_ARCHITECTURES 86
)


