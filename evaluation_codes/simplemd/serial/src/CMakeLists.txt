 

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
target_compile_options(SimpleMD.x PRIVATE
    -O3
)
target_link_libraries(SimpleMD.x PRIVATE SimpleMD::simplemd)
target_compile_features(SimpleMD.x PRIVATE cxx_std_20)




