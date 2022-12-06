/*
 * test/matrix_test.cpp: tests for general matrix operations
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include "matrix.h"

#include <gtest/gtest.h>

#include <iostream>

TEST(MatrixTest, GenerateWithData) {
    int width = 10;
    int height = 10;
    double* h_data;
    h_data = new double[width * height];

    for (int i = 0; i < width * height; ++i) {
        // generate sequential data
        h_data[i] = i;
    }

    Matrix m(h_data, height, width);
    delete[] h_data;

    // get the data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(m.value(i, j), i * width + j);
        }
    }
}

TEST(MatrixTest, GenerateWithoutData) {
    int width = 10;
    int height = 10;
    Matrix m(height, width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(m.value(i, j), m.get_data()[i * width + j]);
            ASSERT_GE(m.value(i, j), 0);
            ASSERT_LE(m.value(i, j), 100);
        }
    }
}

TEST(MatrixTest, Copy) {
    int width = 10;
    int height = 10;
    Matrix m(height, width);

    Matrix m2 = m.copy();

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(m.value(i, j), m2.value(i, j));
        }
    }
}

TEST(MatrixTest, Add) {
    int width = 10;
    int height = 10;
    Matrix m(height, width);
    Matrix m2(height, width);

    Matrix m3 = m + m2;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(m3.value(i, j), m.value(i, j) + m2.value(i, j));
        }
    }
}

TEST(MatrixTest, Sub) {
    int width = 10;
    int height = 10;
    Matrix m(height, width);
    Matrix m2(height, width);

    Matrix m3 = m - m2;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(m3.value(i, j), m.value(i, j) - m2.value(i, j));
        }
    }
}
