/*
 * src/matrix.cpp: Matrix class implementation
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include <cstdlib>
#include "matrix.cuh"
#include "matrix.h"

Matrix::Matrix(double* data, int height, int width) {
    this->data = data;
    this->width = width;
    this->height = height;
}

Matrix::Matrix(int height, int width) {
    this->height = height;
    this->width = width;
    this->data = new double[height * width];
    for (int i = 0; i < height * width; i++) {
        // randomly generate number between 0 and 100
        this->data[i] = rand() % 100;
    }
}

double Matrix::value(int i, int j) {
    return this->data[i * this->width + j];
}

