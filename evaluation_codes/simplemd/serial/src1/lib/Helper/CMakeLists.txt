add_library(helper STATIC)
add_library(SimpleMD::helper ALIAS helper)

# Build the module interface unit
target_sources(helper
  PRIVATE
    FILE_SET cxx_modules TYPE CXX_MODULES
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/Error.ixx
)

# Export headers (so consumers can #include "Helper/Error.hpp")
target_include_directories(helper
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..   # makes src/lib the include root
)

target_compile_features(helper PUBLIC cxx_std_23)





