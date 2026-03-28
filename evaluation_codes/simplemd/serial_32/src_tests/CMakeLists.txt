include(CTest)
include(GoogleTest)

# Recursively collect all .cpp files under src_tests/
file(GLOB_RECURSE TEST_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Create one executable per test source file
foreach(TEST_SRC IN LISTS TEST_FILES)
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_SRC})
    string(REPLACE "/" "_" TEST_NAME "${REL_PATH}")
    string(REPLACE "\\" "_" TEST_NAME "${TEST_NAME}")
    string(REPLACE "." "_"  TEST_NAME "${TEST_NAME}")
  
    set(TEST_TARGET "${TEST_NAME}.x")

    add_executable(${TEST_TARGET} ${TEST_SRC})
    target_link_libraries(${TEST_TARGET} PRIVATE
      SimpleMD::simplemd
      GTest::gtest_main
    )
    gtest_discover_tests(${TEST_TARGET})
endforeach()