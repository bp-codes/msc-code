
add_subdirectory(Helper)

add_library(simplemd INTERFACE)
add_library(SimpleMD::simplemd ALIAS simplemd)

target_include_directories(simplemd
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}          # src/lib (so #include "Helper/Error.hpp")
    ${CMAKE_SOURCE_DIR}/external/json    # so #include <json.hpp>
)

# Make anyone linking SimpleMD::simplemd also link the module library
target_link_libraries(simplemd
  INTERFACE
    SimpleMD::helper
)

