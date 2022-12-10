/*
 * src/matrix.cuh: all matrix operations based on CUDA
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#ifndef MATRIX_CUH
#define MATRIX_CUH
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "matrix.h"

__global__ void GPU_add(double *A, double *B, double *C, int n);
__global__ void GPU_sub(double *A, double *B, double *C, int n);
__global__ void GPU_dot_product(double *A, double *B, double *C, int n);
__global__ void GPU_transpose(double *A, double *B, int width, int height, int start, int step);
__global__ void GPU_multiply(double *A, double *B, double *C, int width, int height);

#endif
