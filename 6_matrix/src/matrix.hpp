#pragma once

#include "config.hpp"

template <size_t N, size_t M>
struct Matrix {
  __align_avx__ double data[N][M];
  inline double* operator[] (size_t i) { return data[i]; }
  inline double const* operator[] (size_t i) const { return data[i]; }
};


using MatrixA = Matrix<MATRIX_SIZE, MATRIX_SIZE>;

