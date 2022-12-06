/*
 * src/matrix.cpp: Matrix class implementation
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include "matrix.cuh"

#include <time.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "matrix.h"

Matrix::Matrix(double* data, int height, int width) {
    double* my_data;

    // judge whether the data is valid
    if (data == NULL) {
        fprintf(stderr, "Error: data is NULL\\n");
        exit(1);
    }
    cudaHostAlloc(&my_data, height * width * sizeof(double),
                  cudaHostAllocDefault);

    // copy the data
    memcpy(my_data, data, height * width * sizeof(double));

    this->data = my_data;
    this->width = width;
    this->height = height;
}

Matrix::Matrix(int height, int width) {
    double* my_data;
    cudaHostAlloc(&my_data, height * width * sizeof(double),
                  cudaHostAllocDefault);

    // initialize the data
    memset(my_data, 0, height * width * sizeof(double));

    this->data = my_data;
    this->width = width;
    this->height = height;
}

Matrix::Matrix(int height, int width, uint urand) {
    this->height = height;
    this->width = width;
    srand(urand);
    cudaHostAlloc(&this->data, height * width * sizeof(double),
                  cudaHostAllocDefault);
    for (int i = 0; i < height * width; i++) {
        // randomly generate number between 0 and 100
        this->data[i] = rand() % 100;
    }
}


Matrix::~Matrix() { cudaFree(this->data); }

double Matrix::value(int i, int j) const {
    return this->data[i * this->width + j];
}

Matrix Matrix::copy() {
    double* new_data;
    cudaHostAlloc(&new_data, this->height * this->width * sizeof(double),
                  cudaHostAllocDefault);
    memcpy(new_data, this->data, this->height * this->width * sizeof(double));
    Matrix m = Matrix(new_data, this->width, this->height);
    cudaFree(new_data);
    return m;
}

Matrix Matrix::operator+(const Matrix& other) {
    double* data;
    cudaHostAlloc(&data, this->height * this->width * sizeof(double),
                  cudaHostAllocDefault);
    for (int i = 0; i < this->height * this->width; i++) {
        data[i] = this->data[i] + other.data[i];
    }
    Matrix m = Matrix(data, this->width, this->height);
    cudaFree(data);
    return m;
}

Matrix Matrix::operator-(const Matrix& other) {
    double* data;
    cudaHostAlloc(&data, this->height * this->width * sizeof(double),
                  cudaHostAllocDefault);
    for (int i = 0; i < this->height * this->width; i++) {
        data[i] = this->data[i] - other.data[i];
    }
    Matrix m = Matrix(data, this->width, this->height);
    cudaFree(data);
    return m;
}

// element-wise multiplication
Matrix Matrix::dot_product(const Matrix& other) {
    double* data;
    cudaHostAlloc(&data, this->height * this->width * sizeof(double),
                  cudaHostAllocDefault);
    for (int i = 0; i < this->height * this->width; i++) {
        data[i] = this->data[i] * other.data[i];
    }

    Matrix m = Matrix(data, this->width, this->height);
    cudaFree(data);
    return m;
}

Matrix Matrix::operator*(const Matrix& other) {
    double* data;
    cudaHostAlloc(&data, this->height * other.width * sizeof(double),
                  cudaHostAllocDefault);

    // check the dimensions
    if (this->width != other.height && this->height != other.width) {
        fprintf(stderr, "Error: matrix dimensions do not match\\n");
        exit(1);
    }

    // do the multiplication
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < other.width; j++) {
            double sum = 0;
            for (int k = 0; k < this->width; k++) {
                sum += this->data[i * this->width + k] *
                       other.data[k * other.width + j];
            }
            data[i * other.width + j] = sum;
        }
    }

    Matrix m = Matrix(data, this->width, this->height);
    cudaFree(data);
    return m;
}

double Matrix::determinant() {
    if (this->height != this->width) {
        fprintf(stderr, "Error: matrix is not square\\n");
        exit(1);
    }
    if (this->height == 1) {
        return this->data[0];
    }
    double det = 0;
    for (int i = 0; i < this->width; i++) {
        Matrix sub_matrix(this->height - 1, this->width - 1);
        for (int j = 0; j < this->height - 1; j++) {
            for (int k = 0; k < this->width - 1; k++) {
                if (k < i) {
                    sub_matrix.data[j * (this->width - 1) + k] =
                        this->data[(j + 1) * this->width + k];
                } else {
                    sub_matrix.data[j * (this->width - 1) + k] =
                        this->data[(j + 1) * this->width + k + 1];
                }
            }
        }
        det += this->data[i] * sub_matrix.determinant() * (i % 2 == 0 ? 1 : -1);
    }
    return det;
}

Matrix Matrix::transpose() {
    double* data;
    cudaHostAlloc(&data, this->height * this->width * sizeof(double),
                  cudaHostAllocDefault);
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            data[j * this->height + i] = this->data[i * this->width + j];
        }
    }
    Matrix m = Matrix(data, this->width, this->height);
    cudaFree(data);
    return m;
}
