/*
 * src/matrix_add_sub.cu: Matrix addition and subtraction
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include <iostream>

#include "matrix.cuh"
#include "matrix.h"

__global__ void GPU_add(double* A, double* B, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }

    return;
}

__global__ void GPU_sub(double* A, double* B, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] - B[i];
    }

    return;
}


Matrix Matrix::gpu_add(const Matrix &other) {
    dim3 dimBlock(1024);
    dim3 dimGrid(((this->height * this -> width) + dimBlock.x - 1) / dimBlock.x);

    Matrix result(this->height, this->width);

    // malloc device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, this->height * this->width * sizeof(double));
    cudaMalloc((void **) &d_B, this->height * this->width * sizeof(double));
    cudaMalloc((void **) &d_C, this->height * this->width * sizeof(double));

    // copy data to device
    cudaMemcpy(d_A, this->data, this->height * this->width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, other.data, this->height * this->width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, result.data, this->height * this->width * sizeof(double), cudaMemcpyHostToDevice);

    GPU_add<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, this->height * this->width);

    // copy data back to host
    cudaMemcpy(result.data, d_C, this->height * this->width * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);

    return result;
}

Matrix Matrix::gpu_sub(const Matrix &other) {
    dim3 dimBlock(1024);
    dim3 dimGrid(((this->height * this -> width) + dimBlock.x - 1) / dimBlock.x);

    Matrix result(this->height, this->width);

    // malloc device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, this->height * this->width * sizeof(double));
    cudaMalloc((void **) &d_B, this->height * this->width * sizeof(double));
    cudaMalloc((void **) &d_C, this->height * this->width * sizeof(double));

    // copy data to device
    cudaMemcpy(d_A, this->data, this->height * this->width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, other.data, this->height * this->width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, result.data, this->height * this->width * sizeof(double), cudaMemcpyHostToDevice);

    GPU_sub<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, this->height * this->width);

    // copy data back to host
    cudaMemcpy(result.data, d_C, this->height * this->width * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);

    return result;
}

