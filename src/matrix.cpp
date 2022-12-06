/*
 * src/matrix.cpp: Matrix class implementation
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include "matrix.cuh"

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
    this->height = height;
    this->width = width;
    cudaHostAlloc(&this->data, height * width * sizeof(double),
                  cudaHostAllocDefault);
    for (int i = 0; i < height * width; i++) {
        // randomly generate number between 0 and 100
        this->data[i] = rand() % 100;
    }
}

Matrix::~Matrix() { cudaFree(this->data); }

double Matrix::value(int i, int j) { return this->data[i * this->width + j]; }

Matrix Matrix::copy() {
    double* new_data;
    cudaHostAlloc(&new_data, this->height * this->width * sizeof(double),
                  cudaHostAllocDefault);
    memcpy(new_data, this->data, this->height * this->width * sizeof(double));
    return Matrix(new_data, this->height, this->width);
}

Matrix Matrix::operator+(const Matrix& other) {
    double* data;
    cudaHostAlloc(&data, this->height * this->width * sizeof(double),
                  cudaHostAllocDefault);
    for (int i = 0; i < this->height * this->width; i++) {
        data[i] = this->data[i] + other.data[i];
    }
    return Matrix(data, this->height, this->width);
}

Matrix Matrix::operator-(const Matrix& other) {
    double* data;
    cudaHostAlloc(&data, this->height * this->width * sizeof(double),
                  cudaHostAllocDefault);
    for (int i = 0; i < this->height * this->width; i++) {
        data[i] = this->data[i] - other.data[i];
    }
    return Matrix(data, this->height, this->width);
}
