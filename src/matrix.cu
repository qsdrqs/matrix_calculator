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
                              int start, int step) {
    int relative_index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = relative_index + start;

    if (relative_index >= step) {
        // out of range
        return;
    }

    int ix = i % width;
    int iy = i / width;

    if (ix < width && iy < height) {
        B[ix * height + iy] = A[i];
    }

    return;
}

__global__ void GPU_multiply(double *A, double *B, double *C, int this_height,
                             int share, int other_width) {
    // use shared memory
    // We assume that a tile is 32 x 32 flattern array
    __shared__ double sA[32][32];
    __shared__ double sB[32][32];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    // now we are going to calculate the C(x, y)
    double sum = 0;

    for (int base = 0; base < share / 32; base++) {
        sA[threadIdx.y][threadIdx.x] = A[y * share + base * 32 + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] =
            B[(base * 32 + threadIdx.y) * other_width + x];

        // wait for all threads to finish
        __syncthreads();

        for (int j = 0; j < 32; j++) {
            sum += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }

        // before move to next tile, wait for all threads to finish
        __syncthreads();
    }

    // the last tile might not be 32 x 32
    if (share % 32 != 0) {
        sA[threadIdx.y][threadIdx.x] = A[y * share + (share / 32) * 32 + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] =
            B[((share / 32) * 32 + threadIdx.y) * other_width + x];

        // wait for all threads to finish
        __syncthreads();

        for (int j = 0; j < share % 32; j++) {
            sum += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }

        // before move to next tile, wait for all threads to finish
        __syncthreads();
    }

    if (x < other_width && y < this_height) {
        C[y * other_width + x] = sum;
    }

    return;
}

Matrix Matrix::gpu_add(const Matrix &other) {
    // check the size of matrices
    if (this->width != other.width || this->height != other.height) {
        fprintf(stderr, "Error: matrix dimensions do not match\n");
    }

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
    // check the size of matrices
    if (this->width != other.width || this->height != other.height) {
        fprintf(stderr, "Error: matrix dimensions do not match\n");
    }
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
    // check the size of matrices
    if (this->width != other.width || this->height != other.height) {
        fprintf(stderr, "Error: matrix dimensions do not match\n");
    }
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
    // define the stream number
    int stream_num = 4;
    dim3 dimBlock(THREADS_PER_BLOCK);
    int size = this->width * this->height;
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);
    dimGrid.x = (dimGrid.x + stream_num - 1) / stream_num;
    Matrix result(this->width, this->height);

    // malloc device memory
    double *d_A, *d_B;
    cudaMalloc((void **)&d_A, size * sizeof(double));
    cudaMalloc((void **)&d_B, size * sizeof(double));

    // implement transpose by stream_num streams
    cudaStream_t stream[stream_num];
    cudaEvent_t event[stream_num];
    for (int i = 0; i < stream_num; i++) {
        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&event[i]);
    }

    int step = (size + stream_num - 1) / stream_num;
    for (int i = 0; i < stream_num; i++) {
        int base = i * step;
        cudaMemcpyAsync(d_A + base, this->data + base,
                        step * sizeof(double), cudaMemcpyHostToDevice,
                        stream[i]);

        GPU_transpose<<<dimGrid, dimBlock, 0, stream[i]>>>(
            d_A, d_B, this->width, this->height, base, step);
        cudaEventRecord(event[i], stream[i]);
    }

    for (int i = 0; i < stream_num; ++i) {
        int base = i * step;
        // wait all other streams and copy data back
        for (int j = 0; j < stream_num; j++) {
            if (j != i) {
                cudaStreamWaitEvent(stream[i], event[j], 0);
            }
        }
        cudaMemcpyAsync(result.data + base, d_B + base,
                        step * sizeof(double), cudaMemcpyDeviceToHost,
                        stream[i]);
    }

    // sync and free streams
    for (int i = 0; i < stream_num; i++) {
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    return result;
}

Matrix Matrix::gpu_multiply(const Matrix &other) {
    // check if the matrix can be multiplied
    if (this->width != other.height) {
        fprintf(stderr, "Error: matrix dimensions do not match\n");
        exit(1);
    }
    dim3 dimBlock(32, 32);
    int this_size = this->width * this->height;
    int other_size = other.width * other.height;
    int result_size = this->height * other.width;
    dim3 dimGrid((other.width + dimBlock.x - 1) / dimBlock.x,
                 (this->height + dimBlock.y - 1) / dimBlock.y);

    Matrix result(this->height, other.width);

    // malloc device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, this_size * sizeof(double));
    cudaMalloc((void **)&d_B, other_size * sizeof(double));
    cudaMalloc((void **)&d_C, result_size * sizeof(double));

    // copy data to device
    cudaMemcpy(d_A, this->data, this_size * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, other.data, other_size * sizeof(double),
               cudaMemcpyHostToDevice);

    GPU_multiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, height, width,
                                        other.width);

    // copy data back to host
    cudaMemcpy(result.data, d_C, result_size * sizeof(double),
               cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}
