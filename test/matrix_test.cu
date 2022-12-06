/*
 * test/matrix_test.cpp: tests for general matrix operations in GPU
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include "matrix.h"

#include <gtest/gtest.h>

#include <iostream>

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
    // test matrix addition
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
