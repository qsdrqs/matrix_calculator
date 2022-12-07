/*
 * src/matrix.h: basic definitions of matrix
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include <cstdlib>

#ifndef MATRIX_H
#define MATRIX_H
class Matrix {
   private:
    int height;
    int width;
    double* data;  // flatten the matrix

   public:
    Matrix(double* data, int height, int width);

    // generate zero matrix
    Matrix(int height, int width);

    // randomly generate matrix
    Matrix(int height, int width, uint urand);

    // from file
    Matrix(const char* filename);

    ~Matrix();

    // get the value of the matrix at (i, j)
    double value(int i, int j) const;

    // getter functions
    double* get_data() const { return this->data; }
    int get_height() const { return this->height; }
    int get_width() const { return this->width; }

    Matrix copy();

    // basic matrix operations by CPU
    Matrix operator+(const Matrix& other);
    Matrix operator-(const Matrix& other);
    Matrix dot_product(const Matrix& other);  // element-wise multiplication
    Matrix operator*(const Matrix& other);

    double determinant();
    Matrix transpose();

    // matrix operations by GPU
    Matrix gpu_add(const Matrix& other);
    Matrix gpu_sub(const Matrix& other);
    Matrix gpu_dot_product(const Matrix& other);  // element-wise multiplication
    Matrix gpu_mul(const Matrix& other);

    double gpu_determinant();
    Matrix gpu_transpose();

    // helper functions
    void print();
    void to_file(const char* filename);
};

#endif
