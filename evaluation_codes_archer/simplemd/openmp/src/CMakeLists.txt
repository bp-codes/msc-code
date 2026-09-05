 
# App binary
add_executable(SimpleMD.x main.cpp)

target_include_directories(SimpleMD.x
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include     
    SYSTEM PRIVATE  
        ${CMAKE_SOURCE_DIR}/external/json     # NLohmann JSON
)

target_link_libraries(SimpleMD.x
    PRIVATE
        OpenMP::OpenMP_CXX
)

target_compile_options(SimpleMD.x PRIVATE
    -O3
    -v
    -ffast-math
)

target_link_options(SimpleMD.x PRIVATE
    -O3
    -v 
    -ffast-math
)
