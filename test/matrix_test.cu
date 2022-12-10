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
    int width = 1024;
    int height = 2048;

    Matrix m(height, width, time(NULL));
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            m.set_value(i, j, i * width + j + 1);
        }
    }

    Matrix m2 = m.transpose();

    Matrix m3 = m.gpu_transpose();

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            EXPECT_EQ(m2.value(i, j), m3.value(i, j));
        }
    }
}

TEST(MatrixGPUTest, Transpose2) {
    // test matrix transpose
    int width = 1000;
    int height = 2000;

    Matrix m(height, width, time(NULL));

    Matrix m2 = m.transpose();

    Matrix m3 = m.gpu_transpose();

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            EXPECT_EQ(m2.value(i, j), m3.value(i, j));
        }
    }
}

TEST(MatrixGPUTest, Transpose3) {
    // test matrix transpose
    int width = 1;
    int height = 2;

    Matrix m(height, width, time(NULL));

    Matrix m2 = m.transpose();

    Matrix m3 = m.gpu_transpose();

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            EXPECT_EQ(m2.value(i, j), m3.value(i, j));
        }
    }
}

TEST(MatrixGPUTest, Multiply) {
    // test matrix multiplication
    int self_height = 64;
    int share = 32;
    int other_width = 64;
    Matrix m(self_height, share, time(NULL));
    Matrix m2(share, other_width, time(NULL));

    Matrix m3 = m * m2;
    Matrix m4 = m.gpu_multiply(m2);

    for (int i = 0; i < self_height; ++i) {
        for (int j = 0; j < other_width; ++j) {
            EXPECT_EQ(m3.value(i, j), m4.value(i, j));
        }
    }
}

TEST(MatrixGPUTest, Multiply2) {
    // test matrix multiplication
    int self_height = 1;
    int share = 2;
    int other_width = 1;
    Matrix m(self_height, share, time(NULL));
    Matrix m2(share, other_width, time(NULL));


    Matrix m3 = m * m2;
    Matrix m4 = m.gpu_multiply(m2);

    for (int i = 0; i < self_height; ++i) {
        for (int j = 0; j < other_width; ++j) {
            EXPECT_EQ(m3.value(i, j), m4.value(i, j));
        }
    }
}

TEST(MatrixGPUTest, Multiply3) {
    // test matrix multiplication
    int self_height = 23;
    int share = 43;
    int other_width = 41;
    Matrix m(self_height, share, time(NULL));
    Matrix m2(share, other_width, time(NULL));


    Matrix m3 = m * m2;
    Matrix m4 = m.gpu_multiply(m2);

    for (int i = 0; i < self_height; ++i) {
        for (int j = 0; j < other_width; ++j) {
            EXPECT_EQ(m3.value(i, j), m4.value(i, j));
        }
    }
}
