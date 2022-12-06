cmake_minimum_required (VERSION 3.0)
project(matrix_calculator)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_compile_definitions(NDEBUG)

message(${CMAKE_SYSTEM_NAME})
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(WINDOWS TRUE)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(LINUX TRUE)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

if (${CMAKE_CL_64})
  set(ARCH_64 TRUE)
  set(PROCESSOR_ARCH "x64")
  message(64)
endif(${CMAKE_CL_64})
#####################################################
find_package(CUDA 10.1)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lnvToolExt -gencode arch=compute_52,code=sm_52)

if (WINDOWS)
set(CUDA_TOOLKIT_SAMPLES_DIR $ENV{HOME}\\cuda-samples)
message(STATUS "cuda sample dir: ${CUDA_TOOLKIT_SAMPLES_DIR}")
else (WINDOWS)

set(CUDA_TOOLKIT_SAMPLES_DIR ${CUDA_TOOLKIT_ROOT_DIR}/samples)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
endif(WINDOWS)
###################################################################
find_package( NVTX REQUIRED )
#find_package( NVML REQUIRED )

set(INCLUDES ${TOP_DIR}/include)

include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/Common/
)

add_subdirectory (src)
add_subdirectory (test)

# add google test
# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
cmake_policy(SET CMP0135 NEW)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
