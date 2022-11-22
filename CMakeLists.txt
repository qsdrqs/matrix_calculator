cmake_minimum_required (VERSION 3.0)
project(lab2)

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

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
	
set(CUDA_TOOLKIT_SAMPLES_DIR ${CUDA_TOOLKIT_ROOT_DIR}/samples)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
endif(WINDOWS)
###################################################################
find_package( NVTX REQUIRED )
#find_package( NVML REQUIRED )
set(BUILD_THIRDPARTY ON)

add_subdirectory (thirdparty)
add_subdirectory (Src)

