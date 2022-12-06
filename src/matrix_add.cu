/*
 * src/matrix_add.cu: Matrix addition implementation
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include "matrix.cuh"
#include "matrix.h"
#include <iostream>

__global__ void test() {
    printf("Hello World!");
}

void foo() {
    test<<<1, 1>>>();
    cudaDeviceSynchronize();
}
