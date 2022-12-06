enable_testing()

aux_source_directory(. TEST_SOURCE_FILES)
aux_source_directory(../src SOURCES)
list(REMOVE_ITEM SOURCES "../src/main.cpp")
message(STATUS "SOURCE_FILES = ${SOURCES}")

cuda_add_executable(
  matrix_calculator_test
  ${TEST_SOURCE_FILES}
  ${SOURCES}
OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math
DEBUG -Xptxas -v -g -G 

)
target_link_libraries(
  matrix_calculator_test
  GTest::gtest_main
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CURAND_LIBRARY}
  ${NVML_DEBUG_LIBRARIES}
  ${NVTX_DEBUG_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(matrix_calculator_test)

target_include_directories(
  matrix_calculator_test
  PUBLIC
  ${INCLUDES}
)
