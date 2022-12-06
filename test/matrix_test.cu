/*
 * test/matrix_test.cpp: tests for general matrix operations
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include <gtest/gtest.h>
#include "matrix.h"

TEST(MatrixTest, GenerateWithData) {
    int width = 10;
    int height = 10;
    double* h_data;
    cudaHostAlloc((void**)&h_data, width * height * sizeof(double), cudaHostAllocDefault);

    for (int i = 0; i < width * height; ++i) {
        // generate sequential data
        h_data[i] = i;
    }

    Matrix m(h_data, height, width);
    // get the data

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(m.value(i, j), i * width + j);
        }
    }

    cudaFreeHost(h_data);
}
