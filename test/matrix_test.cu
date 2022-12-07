/*
 * test/matrix_test.cpp: tests for general matrix operations in GPU
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include <gtest/gtest.h>

#include <iostream>

#include "matrix.h"

TEST(MatrixGPUTest, Add) {
    // test matrix addition
    int width = 10;
    int height = 10;
    Matrix m(height, width, time(NULL));
    Matrix m2(height, width, time(NULL));

    Matrix m3 = m + m2;

    Matrix m4 = m.gpu_add(m2);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(m3.value(i, j), m4.value(i, j));
        }
    }
}

TEST(MatrixGPUTest, Sub) {
    // test matrix subtraction
    int width = 10;
    int height = 10;
    Matrix m(height, width, time(NULL));
    Matrix m2(height, width, time(NULL));

    Matrix m3 = m - m2;

    Matrix m4 = m.gpu_sub(m2);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(m3.value(i, j), m4.value(i, j));
        }
    }
}

TEST(MatrixGPUTest, DotProduct) {
    // test matrix dot product
    int width = 10;
    int height = 10;
    Matrix m(height, width, time(NULL));
    Matrix m2(height, width, time(NULL));

    Matrix m3 = m.dot_product(m2);

    Matrix m4 = m.gpu_dot_product(m2);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(m3.value(i, j), m4.value(i, j));
        }
    }
}

TEST(MatrixGPUTest, Transpose) {
    // test matrix transpose
    int width = 10;
    int height = 20;

    Matrix m(height, width, time(NULL));

    Matrix m2 = m.transpose();

    Matrix m3 = m.gpu_transpose();

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            EXPECT_EQ(m2.value(i, j), m3.value(i, j));
        }
    }
}
