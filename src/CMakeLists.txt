aux_source_directory(. SOURCES)

cuda_add_executable(
  ${PROJECT_NAME}
  ${SOURCES}
OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math
DEBUG -Xptxas -v -g -G 
)

target_include_directories( ${PROJECT_NAME} PUBLIC ${INCLUDES} )
target_link_libraries(
  ${PROJECT_NAME}

  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CURAND_LIBRARY}
  ${NVML_DEBUG_LIBRARIES}
  ${NVTX_DEBUG_LIBRARIES}
)
