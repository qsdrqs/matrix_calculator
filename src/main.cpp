/*
 * src/main.cpp: Main file of matrix calculator
 *
 * Author: Tianyang Zhou <t7zhou@ucsd.edu>
 *
 */

#include <cstring>
#include <iostream>
#include <string>

#include "matrix.cuh"
#include "matrix.h"

void print_usage(char* prog_name) {
    printf(R"(Usage: %s [options] <input files> -o <output file>

Options:
    -h, --help: Print this help message
    -G: Use GPU CUDA to calculate
    --add: add two matrices
    --sub: subtract two matrices
    --mul: multiply two matrices
    --dot: dot product of two matrices
    --random <row_number> <column_number>: generate a random matrix between 0 and 100
    --transpose: transpose a matrix
    --det: calculate the determinant of a matrix
)",
           prog_name);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        if (argc == 2) {
            if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
                print_usage(argv[0]);
                return 0;
            }
            print_usage(argv[0]);
            return 0;
        }
        fprintf(stderr, "Error: too few arguments\n");
        print_usage(argv[0]);
        return 1;
    }

    std::string input_file_1, input_file_2, output_file;

    bool use_gpu = false;
    bool add = false, sub = false, mul = false, dot = false, random = false,
         transpose = false, det = false;
    int random_row = 0, random_col = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-G") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--add") == 0) {
            add = true;
        } else if (strcmp(argv[i], "--sub") == 0) {
            sub = true;
        } else if (strcmp(argv[i], "--mul") == 0) {
            mul = true;
        } else if (strcmp(argv[i], "--dot") == 0) {
            dot = true;
        } else if (strcmp(argv[i], "--random") == 0) {
            random = true;
            // NOTE: This is UNSAFE, but it's OK for this assignment
            random_row = atoi(argv[++i]);
            if (random_row <= 0) {
                fprintf(stderr, "Error: invalid row number for random matrix\n");
                print_usage(argv[0]);
                return 1;
            }
            random_col = atoi(argv[++i]);
            if (random_col <= 0) {
                fprintf(stderr, "Error: invalid column number for random matrix\n");
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "--transpose") == 0) {
            transpose = true;
        } else if (strcmp(argv[i], "--det") == 0) {
            det = true;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                output_file = argv[i + 1];
                ++i;
            } else {
                fprintf(stderr, "Error: No output file specified\n");
                return 1;
            }
        } else if (input_file_1.empty()) {
            input_file_1 = argv[i];
        } else if (input_file_2.empty()) {
            input_file_2 = argv[i];
        }
    }

    if (det) {
        if (use_gpu) {
            fprintf(stderr,
                    "Error: Determinant calculation not supported on GPU\n");
            print_usage(argv[0]);
        } else {
            if (input_file_1.empty()) {
                fprintf(stderr, "Error: No input file specified\n");
                print_usage(argv[0]);
                return 1;
            }
            Matrix mat(input_file_1.c_str());
            std::cout << mat.determinant() << std::endl;
            return 0;
        }
    } else if (output_file.empty()) {
        fprintf(stderr, "Error: No output file specified\n");
        print_usage(argv[0]);
        return 1;
    }


    if (transpose) {
        Matrix m1(input_file_1.c_str());
        if (use_gpu) {
            Matrix m2 = m1.gpu_transpose();
            m2.to_file(output_file.c_str());
            return 0;
        } else {
            Matrix m2 = m1.transpose();
            m2.to_file(output_file.c_str());
            return 0;
        }
    }

    if (random) {
        Matrix m1(random_row, random_col, time(NULL));
        m1.to_file(output_file.c_str());
        return 0;
    }
    if (add || sub || mul || dot) {
        if (input_file_1.empty() || input_file_2.empty()) {
            fprintf(stderr, "Error: Lack of input file\n");
            print_usage(argv[0]);
            return 1;
        }

        Matrix m1(input_file_1.c_str());
        Matrix m2(input_file_2.c_str());

        if (add) {
            if (use_gpu) {
                Matrix m3 = m1.gpu_add(m2);
                m3.to_file(output_file.c_str());
                return 0;
            } else {
                Matrix m3 = m1 + m2;
                m3.to_file(output_file.c_str());
                return 0;
            }
        }

        if (sub) {
            if (use_gpu) {
                Matrix m3 = m1.gpu_sub(m2);
                m3.to_file(output_file.c_str());
                return 0;
            } else {
                Matrix m3 = m1 - m2;
                m3.to_file(output_file.c_str());
                return 0;
            }
        }

        if (mul) {
            if (use_gpu) {
                Matrix m3 = m1.gpu_multiply(m2);
                m3.to_file(output_file.c_str());
                return 0;
            } else {
                Matrix m3 = m1 * m2;
                m3.to_file(output_file.c_str());
                return 0;
            }
        }

        if (dot) {
            if (use_gpu) {
                Matrix m3 = m1.gpu_dot_product(m2);
                m3.to_file(output_file.c_str());
                return 0;
            } else {
                Matrix m3 = m1.dot_product(m2);
                m3.to_file(output_file.c_str());
                return 0;
            }
        }
    }

    return 0;
}
