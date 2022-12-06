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

    ~Matrix();

    // get the value of the matrix at (i, j)
    double value(int i, int j);

    // getter functions
    double* get_data() const { return this->data; }
    int get_height() const { return this->height; }
    int get_width() const { return this->width; }

    Matrix copy();

    // basic matrix operations by CPU
    Matrix operator+(const Matrix& other);
    Matrix operator-(const Matrix& other);
};

#endif
