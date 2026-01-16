 

# Header-only library
add_subdirectory(lib)

add_executable(SimpleMD.x main.cpp)
target_compile_options(SimpleMD.x PRIVATE -O3)
target_compile_features(SimpleMD.x PRIVATE cxx_std_23)

target_link_libraries(SimpleMD.x PRIVATE SimpleMD::simplemd)



