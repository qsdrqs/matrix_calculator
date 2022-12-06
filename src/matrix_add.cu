/*
 * src/matrix_add.cu: Matrix addition implementation
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include <iostream>

#include "matrix.cuh"
#include "matrix.h"

__global__ void test() { printf("Hello World!\n"); }

void foo() {
    test<<<1, 1>>>();
    cudaDeviceSynchronize();
}
