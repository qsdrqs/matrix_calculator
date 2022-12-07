/*
 * src/matrix.cu: Matrix addition and subtraction
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include <iostream>

#include "matrix.cuh"
#include "matrix.h"

#define THREADS_PER_BLOCK 1024

__global__ void GPU_add(double *A, double *B, double *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }

    return;
}

__global__ void GPU_sub(double *A, double *B, double *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] - B[i];
    }

    return;
}

__global__ void GPU_dot_product(double *A, double *B, double *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] * B[i];
    }

    return;
}

__global__ void GPU_transpose(double *A, double *B, int width, int height,
                              int base) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + base;

    int ix = i % width;
    int iy = i / width;

    if (ix < width && iy < height) {
        B[ix * height + iy] = A[i];
    }

    return;
}

__global__ void GPU_multiply(double *A, double *B, double *C, int n) {
    // use share
    return;
}

Matrix Matrix::gpu_add(const Matrix &other) {
    dim3 dimBlock(THREADS_PER_BLOCK);
    int size = this->width * this->height;
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    Matrix result(this->height, this->width);

    // malloc device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size * sizeof(double));
    cudaMalloc((void **)&d_B, size * sizeof(double));
    cudaMalloc((void **)&d_C, size * sizeof(double));

    // copy data to device
    cudaMemcpy(d_A, this->data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, other.data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, result.data, size * sizeof(double), cudaMemcpyHostToDevice);

    GPU_add<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, size);

    // copy data back to host
    cudaMemcpy(result.data, d_C, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);

    return result;
}

Matrix Matrix::gpu_sub(const Matrix &other) {
    dim3 dimBlock(THREADS_PER_BLOCK);
    int size = this->width * this->height;
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    Matrix result(this->height, this->width);

    // malloc device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size * sizeof(double));
    cudaMalloc((void **)&d_B, size * sizeof(double));
    cudaMalloc((void **)&d_C, size * sizeof(double));

    // copy data to device
    cudaMemcpy(d_A, this->data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, other.data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, result.data, size * sizeof(double), cudaMemcpyHostToDevice);

    GPU_sub<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, size);

    // copy data back to host
    cudaMemcpy(result.data, d_C, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);

    return result;
}

Matrix Matrix::gpu_dot_product(const Matrix &other) {
    dim3 dimBlock(THREADS_PER_BLOCK);
    int size = this->width * this->height;
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    Matrix result(this->height, this->width);

    // malloc device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size * sizeof(double));
    cudaMalloc((void **)&d_B, size * sizeof(double));
    cudaMalloc((void **)&d_C, size * sizeof(double));

    // copy data to device
    cudaMemcpy(d_A, this->data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, other.data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, result.data, size * sizeof(double), cudaMemcpyHostToDevice);

    GPU_dot_product<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, size);

    // copy data back to host
    cudaMemcpy(result.data, d_C, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);

    return result;
}

Matrix Matrix::gpu_transpose() {
    dim3 dimBlock(THREADS_PER_BLOCK);
    int size = this->width * this->height;
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);
    Matrix result(this->width, this->height);

    // malloc device memory
    double *d_A, *d_B;
    cudaMalloc((void **)&d_A, size * sizeof(double));
    cudaMalloc((void **)&d_B, size * sizeof(double));

    // implement transpose by 4 streams
    cudaStream_t stream[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&stream[i]);
    }
    for (int i = 0; i < 4; i++) {
        cudaMemcpyAsync(d_A + i * size / 4, this->data + i * size / 4,
                        size * sizeof(double), cudaMemcpyHostToDevice,
                        stream[i]);

        GPU_transpose<<<dimGrid, dimBlock.x / 4, 0, stream[i]>>>(
            d_A, d_B, this->width, this->height, i * size / 4);

        cudaMemcpyAsync(result.data + i * size / 4, d_B + i * size / 4,
                        size * sizeof(double) / 4, cudaMemcpyDeviceToHost,
                        stream[i]);
    }
    // sync and free streams
    for (int i = 0; i < 4; i++) {
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    return result;
}

Matrix Matrix::gpu_mul(const Matrix &other) {
    dim3 dimBlock(THREADS_PER_BLOCK);
    int size = this->width * this->height;
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    Matrix result(this->height, this->width);

    // malloc device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size * sizeof(double));
    cudaMalloc((void **)&d_B, size * sizeof(double));
    cudaMalloc((void **)&d_C, size * sizeof(double));

    // copy data to device
    cudaMemcpy(d_A, this->data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, other.data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, result.data, size * sizeof(double), cudaMemcpyHostToDevice);

    GPU_multiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, size);
    // TODO:
}
