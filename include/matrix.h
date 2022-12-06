/*
 * src/matrix.h: basic definitions of matrix
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#ifndef MATRIX_H
#define MATRIX_H
class Matrix {
   private:
    int height;
    int width;
    double* data;  // flatten the matrix

   public:
    Matrix(double* data, int height, int width);

    // randomly generate matrix
    Matrix(int length, int width);

    // get the value of the matrix at (i, j)
    double value(int i, int j);
};

#endif
