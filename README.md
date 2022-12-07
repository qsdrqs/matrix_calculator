# Matrix Calculator

basic matrix calculation by Nvidia CUDA. Including addition, subtraction,
multiplication and transpose.

# How to compile

On Linux, run the following command to compile.
```bash
cmake -S . -B build

cmake --build build
```

The compiled binary is at `build/src/matrix_calculator`

# How to run test

After successful compiling, run `./build/test/matrix_calculator_test` to run all
tests.

